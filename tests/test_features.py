"""src/strategy/features.py 단위 테스트.

compute_features, compute_multi_tf_features, get_feature_names 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.features import (
    BASE_FEATURE_NAMES,
    compute_features,
    compute_multi_tf_features,
    get_feature_names,
)


def _make_candles(n: int = 300, start_price: float = 67000.0) -> pd.DataFrame:
    """합성 OHLCV DataFrame 생성."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(10, 100, n)
    low = close - rng.uniform(10, 100, n)
    open_ = close + rng.normal(0, 30, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestComputeFeatures:
    def test_column_count(self):
        df = _make_candles(300)
        feat = compute_features(df)
        assert feat.shape[1] == 27
        assert list(feat.columns) == BASE_FEATURE_NAMES

    def test_index_preserved(self):
        df = _make_candles(300)
        feat = compute_features(df)
        assert len(feat) == len(df)
        assert feat.index.equals(df.index)

    def test_nan_in_early_rows(self):
        """EMA200 등 긴 기간 지표로 인해 초기 행은 NaN."""
        df = _make_candles(300)
        feat = compute_features(df)
        assert feat.iloc[0].isna().any()

    def test_valid_rows_after_warmup(self):
        """충분한 워밍업 후 NaN 없는 행 존재."""
        df = _make_candles(300)
        feat = compute_features(df)
        valid = feat.dropna()
        assert len(valid) > 0

    def test_short_data_no_crash(self):
        """데이터가 짧아도 crash하지 않음 (NaN만 나옴)."""
        df = _make_candles(10)
        feat = compute_features(df)
        assert feat.shape[1] == 27


class TestMultiTfFeatures:
    def test_multi_tf_column_count(self):
        candles = {
            "15m": _make_candles(300),
            "1h": _make_candles(75),
        }
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        # 15m: 27 + 1h: 27 = 54
        assert feat.shape[1] == 54

    def test_multi_tf_prefix(self):
        candles = {
            "15m": _make_candles(300),
            "1h": _make_candles(75),
        }
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        prefixed = [c for c in feat.columns if c.startswith("1h_")]
        assert len(prefixed) == 27

    def test_entry_tf_only(self):
        """entry_tf만 있으면 단일 TF와 동일."""
        candles = {"15m": _make_candles(300)}
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        assert feat.shape[1] == 27


class TestGetFeatureNames:
    def test_single_tf(self):
        names = get_feature_names("15m")
        assert len(names) == 27

    def test_multi_tf(self):
        names = get_feature_names("15m", ["1h", "4h"])
        assert len(names) == 27 * 3  # 15m + 1h + 4h

    def test_no_duplicate_entry_tf(self):
        """entry_tf가 extra에 포함되어도 중복 안 됨."""
        names = get_feature_names("15m", ["15m", "1h"])
        assert len(names) == 27 * 2  # 15m + 1h


# ─── I-BLE012: 멀티TF 병합 causal (lookahead 제거) 검증 ───


class TestMultiTfCausalMerge:
    """compute_multi_tf_features가 상위TF를 '봉 마감 시각' 기준 causal 병합 검증.

    I-BLE012 fix: 상위TF 봉을 마감 시각(시작+TF_MS)으로 shift 후 ffill → 각 entry
    시점에 '이미 마감된' 상위봉만 병합. 전체/부분 계산 무관하게 동일(미래 의존 없음).
    기존 now_ms/_is_in_progress_bar 방식은 백테/학습에서 lookahead 였음.
    """

    def test_in_progress_sub_tf_not_merged(self):
        """진행 중(미마감) 상위TF 봉의 완성값이 entry 봉에 병합 안 됨 (lookahead 회귀 방지)."""
        # 15m entry: 00:00~04:45. 1h: 00:00~04:00 (04:00봉은 04:45 시점 진행 중, 마감 05:00)
        entry_dates = pd.date_range("2026-05-06 00:00", "2026-05-06 04:45", freq="15min", tz="UTC")
        entry_df = _make_candles_for_index(entry_dates)
        sub_dates = pd.date_range("2026-05-06 00:00", "2026-05-06 04:00", freq="1h", tz="UTC")
        sub_df = _make_candles_for_index(sub_dates)
        # 마지막 1h 봉(04:00, 마감 05:00)을 극단값으로 — 누출 시 즉시 탐지
        sub_df.iloc[-1] = {"open": 1e9, "high": 1e9, "low": 1e9, "close": 1e9, "volume": 1e9}

        feat = compute_multi_tf_features({"15m": entry_df, "1h": sub_df}, "15m")
        # 04:00 1h봉 마감 = 05:00 → entry 최대 04:45 라 어떤 entry행에도 미병합 (causal)
        col = [c for c in feat.columns if c.startswith("1h_return_")][0]
        assert abs(feat[col].iloc[-1]) < 100, (
            f"진행 중 상위봉(1e9) 완성값이 누출됨(lookahead): {feat[col].iloc[-1]}"
        )

    def test_causal_no_lookahead_full_eq_window(self):
        """전체 계산 == 부분 window 계산 (같은 시점) → 미래 의존 없음(causal) 확정."""
        full_entry = pd.date_range("2026-05-06 00:00", "2026-05-06 08:00", freq="15min", tz="UTC")
        full_sub = pd.date_range("2026-05-06 00:00", "2026-05-06 08:00", freq="1h", tz="UTC")
        candles_full = {
            "15m": _make_candles_for_index(full_entry),
            "1h": _make_candles_for_index(full_sub),
        }
        feat_full = compute_multi_tf_features(candles_full, "15m")

        # window: 05:10 까지만 (라이브가 그 시점에 보는 데이터) — 같은 데이터 slice
        cut = pd.Timestamp("2026-05-06 05:10", tz="UTC")
        candles_win = {tf: df[df.index <= cut] for tf, df in candles_full.items()}
        feat_win = compute_multi_tf_features(candles_win, "15m")

        # 공통 시점(05:00 entry행)의 1h 컬럼이 동일해야 causal
        T = pd.Timestamp("2026-05-06 05:00", tz="UTC")
        for c in [c for c in feat_full.columns if c.startswith("1h_")]:
            a, b = feat_full.loc[T, c], feat_win.loc[T, c]
            if pd.notna(a) and pd.notna(b):
                assert abs(a - b) < 1e-9, f"{c}: full={a} != window={b} (lookahead)"

    def test_entry_rows_share_same_closed_higher_bar(self):
        """같은 상위봉 구간 내 entry행들은 동일 상위TF값 (직전 마감봉 ffill, 시각 정합)."""
        entry_dates = pd.date_range("2026-05-06 00:00", "2026-05-06 06:00", freq="15min", tz="UTC")
        sub_dates = pd.date_range("2026-05-06 00:00", "2026-05-06 06:00", freq="1h", tz="UTC")
        feat = compute_multi_tf_features(
            {"15m": _make_candles_for_index(entry_dates), "1h": _make_candles_for_index(sub_dates)},
            "15m",
        )
        # 03:00~03:45 entry행은 모두 직전 마감 1h봉(02:00, 03:00 마감) 값 → 동일
        # 1h_return_1 은 해당 구간에서 valid. 같은 상위봉 구간 entry행은 동일,
        # 새 상위봉 경계(04:00)에서 값 변경 → causal ffill 시각 정합 검증
        col = "1h_return_1"
        v = feat.loc[pd.Timestamp("2026-05-06 03:15", tz="UTC"), col]
        assert feat.loc[pd.Timestamp("2026-05-06 03:30", tz="UTC"), col] == v
        assert feat.loc[pd.Timestamp("2026-05-06 03:45", tz="UTC"), col] == v
        assert feat.loc[pd.Timestamp("2026-05-06 04:00", tz="UTC"), col] != v

    def test_column_count_unchanged(self):
        """fix 후에도 멀티TF 컬럼 수 불변 (15m+1h = 54)."""
        candles = {"15m": _make_candles(300), "1h": _make_candles(75)}
        feat = compute_multi_tf_features(candles, "15m")
        assert feat.shape[1] == 54


# ─── I-BL007 Phase 3-C: dropna helper 검증 ───


class TestGetCleanLastRow:
    def test_no_nan_returns_last_row_gap_0(self):
        from src.strategy.features import get_clean_last_row
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
            index=pd.date_range("2026-05-06", periods=3, freq="15min", tz="UTC"),
        )
        row, diag = get_clean_last_row(df, ["a", "b"])
        assert row is not None
        assert list(row) == [3.0, 6.0]
        assert diag["gap_to_latest"] == 0  # 가장 최근 봉 사용

    def test_nan_in_last_row_gap_1(self):
        """마지막 row가 NaN이면 직전 row 사용 + gap=1 (진행 중 봉 제외 시나리오)."""
        from src.strategy.features import get_clean_last_row
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]},
            index=pd.date_range("2026-05-06", periods=3, freq="15min", tz="UTC"),
        )
        row, diag = get_clean_last_row(df, ["a", "b"])
        assert row is not None
        assert list(row) == [2.0, 5.0]  # 직전 row
        assert diag["gap_to_latest"] == 1

    def test_long_indicator_nan_in_early_rows_gap_0(self):
        """앞쪽 row가 NaN(예: 200 EMA)이고 마지막 row valid → gap=0 (정상)."""
        from src.strategy.features import get_clean_last_row
        df = pd.DataFrame(
            {"a": [np.nan] * 200 + [1.0, 2.0, 3.0], "b": [np.nan] * 200 + [4.0, 5.0, 6.0]},
            index=pd.date_range("2026-05-06", periods=203, freq="15min", tz="UTC"),
        )
        row, diag = get_clean_last_row(df, ["a", "b"])
        assert row is not None
        assert list(row) == [3.0, 6.0]
        # long indicator NaN은 used_ts 이전이라 gap에 카운트 안 됨
        assert diag["gap_to_latest"] == 0

    def test_all_nan_returns_none_with_nan_by_tf(self):
        from src.strategy.features import get_clean_last_row
        df = pd.DataFrame(
            {
                "1h_body_ratio": [np.nan, np.nan],
                "1h_upper_shadow": [np.nan, np.nan],
                "4h_atr_pct": [np.nan, np.nan],
                "rsi_14": [50.0, np.nan],
            },
            index=pd.date_range("2026-05-06", periods=2, freq="15min", tz="UTC"),
        )
        row, diag = get_clean_last_row(
            df, ["1h_body_ratio", "1h_upper_shadow", "4h_atr_pct", "rsi_14"]
        )
        assert row is None
        assert diag["fail_reason"] == "all_features_nan"
        assert diag["nan_by_tf"]["1h"] == ["body_ratio", "upper_shadow"]
        assert diag["nan_by_tf"]["4h"] == ["atr_pct"]
        assert diag["nan_by_tf"]["entry_tf"] == ["rsi_14"]


class TestGetCleanFeaturesForSequence:
    def test_enough_rows_no_gap(self):
        from src.strategy.features import get_clean_features_for_sequence
        df = pd.DataFrame(
            {"a": list(range(70)), "b": list(range(70))},
            index=pd.date_range("2026-05-06", periods=70, freq="15min", tz="UTC"),
        )
        clean, diag = get_clean_features_for_sequence(df, ["a", "b"], 60)
        assert clean is not None
        assert len(clean) == 70
        assert diag["gap_to_latest"] == 0

    def test_dropna_lt_lookback_returns_none(self):
        from src.strategy.features import get_clean_features_for_sequence
        a = list(range(50)) + [np.nan] * 20  # 마지막 20개 NaN
        df = pd.DataFrame(
            {"a": a, "b": list(range(70))},
            index=pd.date_range("2026-05-06", periods=70, freq="15min", tz="UTC"),
        )
        clean, diag = get_clean_features_for_sequence(df, ["a", "b"], 60)
        assert clean is None
        assert diag["fail_reason"] == "dropna_lt_lookback"
        assert diag["available_rows"] == 50
        assert diag["required_lookback"] == 60

    def test_last_row_nan_gap_1(self):
        """마지막 row NaN + lookback 충분 → gap=1."""
        from src.strategy.features import get_clean_features_for_sequence
        a = list(range(70)) + [np.nan]  # 마지막만 NaN, 70 valid
        df = pd.DataFrame(
            {"a": a, "b": list(range(71))},
            index=pd.date_range("2026-05-06", periods=71, freq="15min", tz="UTC"),
        )
        clean, diag = get_clean_features_for_sequence(df, ["a", "b"], 60)
        assert clean is not None
        assert len(clean) == 70
        assert diag["gap_to_latest"] == 1


class TestGroupNanByTf:
    def test_grouping(self):
        from src.strategy.features import _group_nan_by_tf
        result = _group_nan_by_tf(
            ["1h_body_ratio", "4h_atr_pct", "rsi_14", "1h_macd"]
        )
        assert result["1h"] == ["body_ratio", "macd"]
        assert result["4h"] == ["atr_pct"]
        assert result["entry_tf"] == ["rsi_14"]


# ─── I-BL007 Phase 3-D: get_features_for_ctx 라이브 path도 ts < now 적용 ───


class TestGetFeaturesForCtxLivePath:
    """라이브 path(precomputed_features=None)도 ctx.now 미만 cutoff 적용 검증.

    핵심 fix: 봉 t+1 시작 시점에 _on_bar_closed 호출 → ctx.now=t+1 →
    features.index < t+1 → 마지막 = 봉 t (마감 봉) → 학습/backtest와 동일 cycle.
    """

    def test_live_path_excludes_now_bar(self):
        """라이브에서 ctx.now에 해당하는 봉(진행 중)이 features에서 제외됨."""
        from src.core.types import StrategyContext
        from src.strategy.features import get_features_for_ctx

        # 백필 + 새 봉 시뮬: 마지막 봉 timestamp = 04:45 (방금 시작)
        candles = pd.date_range(
            "2026-05-06 00:00", "2026-05-06 04:45", freq="15min", tz="UTC"
        )
        df = _make_candles_for_index(candles)
        ctx = StrategyContext(
            candles={"15m": df},
            current_price=67000.0,
            balance=10000.0,
            position=None,
            is_slot_occupied=False,
            params={},
            now=pd.Timestamp("2026-05-06 04:45", tz="UTC").to_pydatetime(),
            precomputed_features=None,  # 라이브 모드
        )
        features = get_features_for_ctx(ctx, "15m")
        # 마지막 row의 timestamp가 ctx.now 미만이어야 함
        assert (features.index < ctx.now).all()
        # 마지막 봉(04:45)은 features에 없음
        assert pd.Timestamp("2026-05-06 04:45", tz="UTC") not in features.index

    def test_backtest_path_unchanged(self):
        """백테 path는 기존 동작 그대로 (precomputed_features 사용)."""
        from src.core.types import StrategyContext
        from src.strategy.features import get_features_for_ctx

        precomputed = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=pd.date_range("2026-05-06", periods=3, freq="15min", tz="UTC"),
        )
        ctx = StrategyContext(
            candles={},
            current_price=67000.0,
            balance=10000.0,
            position=None,
            is_slot_occupied=False,
            params={},
            now=pd.Timestamp("2026-05-06 00:30", tz="UTC").to_pydatetime(),
            precomputed_features=precomputed,
        )
        features = get_features_for_ctx(ctx, "15m")
        # 00:30 미만만 → 00:00, 00:15
        assert len(features) == 2
        assert (features.index < ctx.now).all()

    def test_live_path_empty_candles_returns_empty(self):
        """빈 candles dict → 빈 DataFrame 반환 (안전 fallback)."""
        from src.core.types import StrategyContext
        from src.strategy.features import get_features_for_ctx

        # 빈 dataframe
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        ctx = StrategyContext(
            candles={"15m": empty_df},
            current_price=67000.0,
            balance=10000.0,
            position=None,
            is_slot_occupied=False,
            params={},
            now=pd.Timestamp("2026-05-06 04:45", tz="UTC").to_pydatetime(),
            precomputed_features=None,
        )
        features = get_features_for_ctx(ctx, "15m")
        assert features.empty


def _make_candles_for_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """특정 인덱스에 합성 OHLCV 생성."""
    rng = np.random.default_rng(42)
    n = len(idx)
    close = 67000.0 + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(10, 100, n)
    low = close - rng.uniform(10, 100, n)
    open_ = close + rng.normal(0, 30, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
