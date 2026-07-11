"""Step 0.1~0.2 컴포넌트 간 seam 통합 검증.

각 컴포넌트 단위 테스트와 별개로, 공유 계약(감사 통과 데이터 → splitter/regime,
완성봉 인과 정렬)이 실제로 맞물리는지 검증한다. 실데이터 의존 테스트는 캔들 파일이
없으면 skip.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.audit import AuditError
from src.research.data.loader import (
    OHLCV_COLUMNS,
    csv_filename,
    load_audited,
    load_ohlcv,
)
from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.ledger import RunLedger
from src.research.validation.metrics import FoldPrediction, aggregate_metrics
from src.research.validation.regime import (
    LABEL_COL,
    RegimeParams,
    forward_fill_completed,
    tag_regimes,
)
from src.research.validation.splitter import WalkForwardSplitter

CANDLE_DIR = "data/candles"
SYM = "BTC/USDT:USDT"


def _has(symbol, tf):
    return os.path.exists(os.path.join(CANDLE_DIR, csv_filename(symbol, tf)))


requires_1h = pytest.mark.skipif(not _has(SYM, "1h"), reason="BTC 1h 캔들 없음")
requires_1d = pytest.mark.skipif(not _has(SYM, "1d"), reason="BTC 1d 캔들 없음")


def _clean_df(idx):
    n = len(idx)
    return pd.DataFrame(
        {"open": [10.0] * n, "high": [11.0] * n, "low": [9.0] * n,
         "close": [10.5] * n, "volume": [1.0] * n},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


# ---- seam: loader → audit (load_audited 계약) ----

def test_load_audited_clean_returns_df(tmp_path):
    idx = pd.date_range("2020-01-01", periods=50, freq="h", tz="UTC")
    _clean_df(idx).to_csv(tmp_path / csv_filename(SYM, "1h"))
    df = load_audited(SYM, "1h", candle_dir=str(tmp_path))
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 50


def test_load_audited_blocks_corruption_at_entry(tmp_path):
    # 오프그리드(16:00) 봉이 섞인 1d → 진입에서 하드페일 (I-001 류를 seam 에서 차단)
    idx = list(pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC"))
    idx.append(idx[2] + pd.Timedelta(hours=16))
    idx = pd.DatetimeIndex(sorted(idx), name="timestamp")
    _clean_df(idx).to_csv(tmp_path / csv_filename(SYM, "1d"))
    with pytest.raises(AuditError):
        load_audited(SYM, "1d", candle_dir=str(tmp_path))


# ---- seam: load_audited → splitter (실 1h) ----

@requires_1h
def test_real_1h_audited_feeds_splitter():
    df = load_audited(SYM, "1h")            # 1h 는 clean → 통과
    sp = WalkForwardSplitter(train_min=8760, val_size=2160, label_horizon=24)
    folds = list(sp.split(df.index))
    assert len(folds) >= 10
    lo, hi = df.index[0], df.index[-1]
    for f in folds:
        assert f.train_index[-1] < f.val_index[0]        # purge 유지
        assert f.train_index[0] >= lo and f.val_index[-1] <= hi  # 범위 내


@requires_1d
def test_real_1d_is_currently_corrupt_I001():
    # I-001 문서화: 현재 1d 는 오프그리드로 load_audited 가 막아야 한다.
    # (remediation 완료 시 이 테스트는 갱신 대상.)
    with pytest.raises(AuditError):
        load_audited(SYM, "1d")


# ---- seam: loader(1d) → regime → forward_fill → 1h 인덱스 (실데이터 인과 정렬) ----

@requires_1h
@requires_1d
def test_real_regime_forward_fill_causal_alignment():
    d1 = load_ohlcv(SYM, "1d")             # 1d 는 감사 미통과라 load_ohlcv 로(태깅은 clean 구간)
    tags = tag_regimes(d1)[LABEL_COL]
    h1_index = load_ohlcv(SYM, "1h").index
    ff = forward_fill_completed(tags, h1_index, "1d")

    # 정렬: 하위 인덱스와 1:1
    assert len(ff) == len(h1_index)
    assert (ff.index == h1_index).all()

    # 인과성: 특정 1h 시각의 1d 국면 = 직전 '완성' 1d 봉의 국면 (진행 중 봉 배제)
    ts = pd.Timestamp("2021-06-15 12:00", tz="UTC")
    prev_completed_day = pd.Timestamp("2021-06-14", tz="UTC")
    if ts in h1_index and prev_completed_day in tags.index:
        assert ff.loc[ts] == tags.loc[prev_completed_day]
        # 진행 중 당일(2021-06-15) 국면을 앞당겨 쓰지 않았는지
        same_day = pd.Timestamp("2021-06-15", tz="UTC")
        if same_day in tags.index and tags.loc[same_day] != tags.loc[prev_completed_day]:
            assert ff.loc[ts] != tags.loc[same_day]


# ---- seam: normalize × splitter Fold (train-only, 규칙 16) ----

def _feat_frame(n):
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    rng = range(n)
    return pd.DataFrame(
        {"f1": [float(i) for i in rng], "f2": [float((i * 7) % 13) for i in rng]},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


def test_normalize_fits_on_fold_train_only():
    feat = _feat_frame(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    fold = next(iter(sp.split(feat.index)))

    norm = ZScoreNormalizer().fit(feat.loc[fold.train_index])
    mean_before = norm.mean_.copy()

    # val 값을 크게 교란해도 train 통계는 불변이어야 함 (fit 이 val 을 안 봄)
    feat_pert = feat.copy()
    feat_pert.loc[fold.val_index] = feat_pert.loc[fold.val_index] * 100 + 1e6
    norm2 = ZScoreNormalizer().fit(feat_pert.loc[fold.train_index])
    assert (norm2.mean_ == mean_before).all()

    # val 변환은 train 통계로 (x - train_mean)/train_std
    val_out = norm.transform(feat.loc[fold.val_index])
    expected = (feat.loc[fold.val_index] - norm.mean_) / norm.std_
    assert val_out.equals(expected)


# ---- seam: metrics × splitter Fold × regime (봉단위 귀속, 규칙 16) ----

def test_metrics_aggregation_over_folds_and_regimes():
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    y = pd.Series(np.array(["up", "down", "flat"])[np.arange(n) % 3], index=idx)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    regime = pd.Series(
        np.where((np.arange(n) // 100) % 2 == 0, "bull", "bear"), index=idx
    )
    labels = sorted(y.unique())

    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    fold_preds = []
    for fold in sp.split(idx):
        m = PriorBaseline().fit(X.loc[fold.train_index], y.loc[fold.train_index])
        proba = m.predict_proba(X.loc[fold.val_index])
        pred = pd.Series(m.predict(X.loc[fold.val_index]), index=fold.val_index)
        fold_preds.append(
            FoldPrediction(fold.fold_id, y.loc[fold.val_index], pred, proba)
        )

    rep = aggregate_metrics(fold_preds, regime_tags=regime, labels=labels)

    # 폴드별 분포 (평균 하나로 안 뭉갬)
    assert len(rep.per_fold) == len(fold_preds)
    assert {"balanced_accuracy", "mcc", "log_loss"} <= set(rep.per_fold.columns)
    assert set(rep.summary.columns) == {"mean", "median", "min", "max", "std"}
    # 국면별 봉단위 귀속
    assert set(rep.per_regime.index) <= {"bull", "bear"}
    assert "n" in rep.per_regime.columns
    # 국면별 n 합 = 전 폴드 검증봉 수
    total_val = sum(f.n_val for f in [fold for fold in sp.split(idx)])
    assert rep.per_regime["n"].sum() == total_val


# ---- seam: 전체 파이프라인 harness (실 1h, 규칙 16) ----

@requires_1h
@requires_1d
def test_real_1h_harness_end_to_end():
    """실 1h full-stack: load_audited → regime(1d ff) → normalize → harness → per-regime."""
    df = load_audited(SYM, "1h")                 # 감사 통과(clean)
    # throwaway 인과 라벨: 다음봉 부호 (N=1) — label 은 forward, splitter 가 purge/꼬리예약
    diff = df["close"].shift(-1) - df["close"]
    y = pd.Series(np.sign(diff.to_numpy()), index=df.index).map(
        {1.0: "up", -1.0: "down", 0.0: "flat"}
    )
    X = pd.DataFrame(
        {
            "r1": df["close"].pct_change(),
            "v20": np.log(df["close"] / df["close"].shift(1)).rolling(20).std(),
        },
        index=df.index,
    )
    # 국면: 1d(load_ohlcv — I-001 꼬리 인지) → 완성봉 ff → 1h
    regime = forward_fill_completed(
        tag_regimes(load_ohlcv(SYM, "1d"))[LABEL_COL], df.index, "1d"
    )
    sp = WalkForwardSplitter(train_min=8760, val_size=2160, label_horizon=1)

    res = run_walk_forward(
        X, y, PriorBaseline, sp,
        regime_tags=regime, labels=["down", "flat", "up"],
        normalizer_factory=ZScoreNormalizer,
    )

    assert res.n_folds >= 10
    assert {"balanced_accuracy", "mcc", "log_loss"} <= set(res.report.per_fold.columns)
    assert not res.report.per_regime.empty          # 국면 봉단위 귀속
    assert res.config["normalized"] is True          # normalize 파이프라인 편입
    # baseline 은 엣지가 없어야 정상: 균형정확도 ~ 1/유효클래스 근처(대략)
    assert res.report.summary.loc["balanced_accuracy", "mean"] < 0.6


# ---- 통짜(whole-Phase) full-stack: 전 컴포넌트 한 번에 (합성 결정적) ----

def _synth_close(n, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    t = np.arange(n)
    return pd.Series(100.0 * (1 + 0.0003 * t + 0.03 * np.sin(t / 40.0)), index=idx)


def _ohlcv(close):
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1.0},
        index=close.index,
    )


def test_full_stack_synthetic(tmp_path):
    """regime + normalize + splitter(purge) + harness + per-regime + ledger 를 한 흐름에서."""
    P = RegimeParams(ma_len=5, slope_k=3, deadband=0.001, vol_len=5,
                     vol_hi_pct=0.5, min_duration=1)
    d1 = _ohlcv(_synth_close(200, "2020-01-01", "D"))       # 1d (regime)
    tags = tag_regimes(d1, P)[LABEL_COL]

    h_close = _synth_close(1500, "2020-02-01", "h")          # 1h (모델 TF, 태깅 구간 내)
    regime = forward_fill_completed(tags, h_close.index, "1d")

    N = 10
    X = pd.DataFrame(
        {"r1": h_close.pct_change(),
         "v5": np.log(h_close / h_close.shift(1)).rolling(5).std()},
        index=h_close.index,
    )
    fwd = h_close.shift(-N) / h_close - 1
    y = pd.Series(index=h_close.index, dtype=object)
    y[fwd > 0.002] = "up"
    y[fwd < -0.002] = "down"
    y[(fwd >= -0.002) & (fwd <= 0.002)] = "flat"

    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=N)
    res = run_walk_forward(
        X, y, PriorBaseline, sp,
        regime_tags=regime, labels=["down", "flat", "up"],
        normalizer_factory=ZScoreNormalizer, seed=7,
    )

    # 전 체인 정합
    assert res.n_folds >= 2
    assert {"balanced_accuracy", "mcc", "log_loss"} <= set(res.report.per_fold.columns)
    assert res.config["normalized"] is True
    assert not res.report.per_regime.empty
    # 봉단위 국면 귀속 완전성: 국면별 n 합 = 전 폴드 검증봉 수
    total_val = sum(
        f.n_val for f in WalkForwardSplitter(300, 100, N).split(h_close.index)
    )
    assert res.report.per_regime["n"].sum() == total_val

    # ledger 편입
    led = RunLedger(str(tmp_path / "ledger.jsonl"))
    led.record({"model": "prior", **res.config}, {"n_folds": res.n_folds},
               campaign="fullstack")
    assert led.comparison_count("fullstack") == 1

    # 결정적 재현
    res2 = run_walk_forward(
        X, y, PriorBaseline, sp,
        regime_tags=regime, labels=["down", "flat", "up"],
        normalizer_factory=ZScoreNormalizer, seed=7,
    )
    assert res.report.per_fold.equals(res2.report.per_fold)
