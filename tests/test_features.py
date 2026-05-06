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


# ─── I-BL007 Phase 3: 진행 중 봉 제외 검증 ───


class TestInProgressBarHelper:
    """_is_in_progress_bar 단위 테스트."""

    def test_bar_not_yet_closed_returns_true(self):
        from src.strategy.features import _is_in_progress_bar
        # 1h 봉 04:00 시작 (마감 예정 05:00). now=04:45 → 진행 중
        sub_last = pd.Timestamp("2026-05-06 04:00:00", tz="UTC")
        sub_tf_ms = 3_600_000
        now_ms = int(pd.Timestamp("2026-05-06 04:45:00", tz="UTC").timestamp() * 1000)
        assert _is_in_progress_bar(sub_last, sub_tf_ms, now_ms) is True

    def test_bar_already_closed_returns_false(self):
        from src.strategy.features import _is_in_progress_bar
        # 1h 봉 04:00 시작 (마감 05:00). now=05:30 → 마감 봉
        sub_last = pd.Timestamp("2026-05-06 04:00:00", tz="UTC")
        sub_tf_ms = 3_600_000
        now_ms = int(pd.Timestamp("2026-05-06 05:30:00", tz="UTC").timestamp() * 1000)
        assert _is_in_progress_bar(sub_last, sub_tf_ms, now_ms) is False

    def test_historical_data_always_closed(self):
        """학습 시점 — historical data의 봉은 항상 마감으로 판정 (학습-추론 일관성)."""
        from src.strategy.features import _is_in_progress_bar
        sub_last = pd.Timestamp("2024-12-31 12:00:00", tz="UTC")
        sub_tf_ms = 3_600_000
        # 현재 시각으로 가정 (1년 후)
        now_ms = int(pd.Timestamp("2026-05-06 00:00:00", tz="UTC").timestamp() * 1000)
        assert _is_in_progress_bar(sub_last, sub_tf_ms, now_ms) is False


class TestMultiTfInProgressExclusion:
    """compute_multi_tf_features가 진행 중 sub_tf 봉을 제외 검증."""

    def test_in_progress_sub_tf_excluded(self, monkeypatch):
        """1h 진행 중 봉이 features 계산에서 제외 — 진행 중 봉의 OHLC 영향 차단."""
        # 15m entry_tf 마지막 봉 = 04:45 (= 마감, 다음 봉 05:00 시작 직후)
        # 1h 마지막 봉 = 04:00 (= 진행 중, 마감 예정 05:00)
        # now_ms = 04:46 (15m 봉 마감 직후)
        entry_dates = pd.date_range("2026-05-06 00:00", "2026-05-06 04:45", freq="15min", tz="UTC")
        entry_df = _make_candles_for_index(entry_dates)
        sub_dates_full = pd.date_range("2026-05-06 00:00", "2026-05-06 04:00", freq="1h", tz="UTC")
        sub_df = _make_candles_for_index(sub_dates_full)
        # 마지막 1h 봉(04:00)을 명백히 다른 OHLC로 설정 — 제외 확인용
        sub_df.iloc[-1] = {"open": 1e9, "high": 1e9, "low": 1e9, "close": 1e9, "volume": 1e9}

        # now_ms를 04:46으로 fix
        from src.strategy import features as features_mod
        fake_now_ms = int(pd.Timestamp("2026-05-06 04:46:00", tz="UTC").timestamp() * 1000)
        class _FakeDt:
            @staticmethod
            def now(tz=None):
                return pd.Timestamp("2026-05-06 04:46:00", tz="UTC").to_pydatetime()
        monkeypatch.setattr(features_mod, "datetime", _FakeDt)

        candles = {"15m": entry_df, "1h": sub_df}
        feat = compute_multi_tf_features(candles, "15m")

        # 1h_close 컬럼이 있다면, ffill 후 마지막 row의 1h 컬럼이 직전 마감 봉(03:00)의 features 기반이어야 함.
        # 진행 중 봉(04:00, OHLC=1e9)이 포함됐다면 indicator가 비정상 큰 값을 만들 것
        # 단순 검증: 1h prefix 컬럼이 결과에 존재하고 값이 비정상이지 않음
        col_1h_returns = [c for c in feat.columns if c.startswith("1h_return_")]
        assert len(col_1h_returns) >= 1
        # 1e9 close가 features에 들어왔으면 return 값이 비정상으로 큼
        last_1h_return = feat[col_1h_returns[0]].iloc[-1]
        assert abs(last_1h_return) < 100, (
            f"진행 중 봉(close=1e9) 영향이 features에 누출됨: {last_1h_return}"
        )

    def test_all_bars_closed_no_exclusion(self, monkeypatch):
        """학습 시나리오 — 모든 봉 마감 → 진행 중 판정 없음 → 기존 동작 그대로."""
        entry_dates = pd.date_range("2024-01-01", "2024-01-15", freq="15min", tz="UTC")
        entry_df = _make_candles_for_index(entry_dates)
        sub_dates = pd.date_range("2024-01-01", "2024-01-15", freq="1h", tz="UTC")
        sub_df = _make_candles_for_index(sub_dates)

        # now_ms = 현재 시각 (historical보다 미래)
        from src.strategy import features as features_mod
        class _FakeDt:
            @staticmethod
            def now(tz=None):
                return pd.Timestamp("2026-05-06", tz="UTC").to_pydatetime()
        monkeypatch.setattr(features_mod, "datetime", _FakeDt)

        candles = {"15m": entry_df, "1h": sub_df}
        feat = compute_multi_tf_features(candles, "15m")
        # 결과가 정상 생성되고, 컬럼 수가 27*2 (15m + 1h)
        assert any(c.startswith("1h_") for c in feat.columns)


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
