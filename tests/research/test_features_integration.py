"""Phase 2 피처 ↔ substrate 통합 seam (규칙 16·17) — Step 2.4-a/b/c.

Step 단위테스트(격리)가 못 보는 접합부를 커밋 회귀로 박제:
- **a) X ↔ harness/splitter/normalize**: 8축 X 가 run_walk_forward(정규화+폴드+모델)를
  계약대로 통과(6축 동시 흐름·index 정렬·폴드구간 유한). 라벨↔X index 계약도 확인.
- **b) recursive KF 폴드-슬라이싱 인과**: build_features(df).loc[val] ==
  build_features(df[:val_end]).loc[val] — KF(유일 stateful 축)가 폴드 경계에서 미래 무참조.
- **c) 정규화 후 불확실성 생존**: 폴드 train-only z-score 후 kf_uncertainty 정보 유지.

라벨 NaN 드롭(F-10)은 Prior baseline 이 X·y분포만 쓰고 val 라벨을 clean 으로 두어 미발현 —
Phase 3 실모델 통합 전 해결 대상(계획 유지).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import csv_filename, load_audited
from src.research.features.build import build_features
from src.research.labeling.triple_barrier import LabelParams, compute_triple_barrier
from src.research.normalize import RobustScaler, ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.splitter import WalkForwardSplitter
from tests.research.synthetic import make_ohlcv

SYM = "BTC/USDT:USDT"
CANDLE_DIR = "data/candles"
requires_1h = pytest.mark.skipif(
    not os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, "1h"))),
    reason="BTC 1h 캔들 없음",
)


def _clean_labels(index, seed=1) -> pd.Series:
    """X seam 격리용 clean 3-class y (라벨 NaN=F-10 은 분리)."""
    rng = np.random.default_rng(seed)
    vals = rng.choice(["up", "down", "expire"], size=len(index), p=[0.31, 0.31, 0.38])
    return pd.Series(vals, index=index, name="label")


def _outlier_ohlcv(n=400, seed=0):
    """이상치 클러스터 포함 OHLCV — kf_uncertainty 변동 유발."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.004, n)
    ret[200:230] += rng.choice([-1, 1], 30) * 0.08          # 격변 구간
    close = 10000 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1, 100, n)},
        index=idx,
    )


# ---- 2.4-a: X ↔ harness/splitter/normalize seam ----

def test_features_flow_through_harness():
    """8축 X 가 정규화+폴드+baseline 을 계약대로 통과."""
    df = make_ohlcv(n=800, seed=3)
    X = build_features(df)
    assert X.shape[1] == 8
    y = _clean_labels(df.index)

    sp = WalkForwardSplitter(train_min=200, val_size=100, label_horizon=24, step=100)
    res = run_walk_forward(
        X, y, PriorBaseline, sp,
        normalizer_factory=ZScoreNormalizer, seed=0,
    )
    assert res.n_folds >= 2
    # 각 폴드 val(스코어 구간)은 피처 워밍업 이후 → X 유한
    for fold in sp.split(X.index):
        assert np.isfinite(X.loc[fold.val_index].to_numpy()).all()
    # 집계 리포트가 폴드별 지표를 냄(vacuous 아님)
    assert res.report.per_fold.shape[0] == res.n_folds


def test_features_label_index_contract():
    """X ↔ 삼중배리어 라벨 index 정렬 계약(하류 harness 가 정렬 전제)."""
    df = make_ohlcv(n=800, seed=3)
    X = build_features(df)
    bl = compute_triple_barrier(df, LabelParams("atr", 96, 3.0, 24))
    assert X.index.equals(bl.labels.index)


# ---- 2.4-b: recursive KF 폴드-슬라이싱 인과 ----

def test_fold_slicing_is_causal():
    """폴드 경계에서 전체계산 X 와 val_end 절단계산 X 가 val 에서 일치(미래 무참조)."""
    df = make_ohlcv(n=800, seed=4)
    X_full = build_features(df)
    sp = WalkForwardSplitter(train_min=200, val_size=100, label_horizon=24, step=100)
    folds = list(sp.split(df.index))
    assert len(folds) >= 2
    for fold in folds:
        val_end_pos = df.index.get_indexer([fold.val_index[-1]])[0]
        X_trunc = build_features(df.iloc[: val_end_pos + 1])
        pd.testing.assert_frame_equal(
            X_full.loc[fold.val_index], X_trunc.loc[fold.val_index]
        )


# ---- 2.4-c: 정규화 후 불확실성 생존 ----

def test_kf_uncertainty_survives_fold_normalization():
    """calm train 으로 fit 한 z-score 가 turbulent val 의 불확실성 정보를 안 죽인다."""
    df = _outlier_ohlcv(n=400, seed=0)
    X = build_features(df)
    col = ["kf_uncertainty"]
    # calm 구간(격변 200~230 전) train, 격변 포함 val
    train_idx = X.index[130:190]
    val_idx = X.index[195:245]
    norm = ZScoreNormalizer().fit(X.loc[train_idx, col])
    z_val = norm.transform(X.loc[val_idx, col])["kf_uncertainty"]
    assert np.isfinite(z_val.to_numpy()).all()
    # 격변 val 불확실성이 상수로 뭉개지지 않음(정보 생존)
    assert z_val.std() > 0.1
    # 원시 불확실성이 실제로 변동했음(대조)
    assert X.loc[val_idx, "kf_uncertainty"].std() > 0


# ---- 2.4-d seam: RobustScaler(F-6 신규 컴포넌트) 드롭인 (규칙16) ----

def test_robust_scaler_drops_into_harness():
    """RobustScaler 가 harness normalizer_factory 로 z-score 처럼 통과(train-only)."""
    df = make_ohlcv(n=800, seed=3)
    X = build_features(df)
    y = _clean_labels(df.index)
    sp = WalkForwardSplitter(train_min=200, val_size=100, label_horizon=24, step=100)
    res = run_walk_forward(
        X, y, PriorBaseline, sp,
        normalizer_factory=RobustScaler, seed=0,
    )
    assert res.n_folds >= 2
    assert res.report.per_fold.shape[0] == res.n_folds
    assert res.config["normalized"] is True


def test_robust_scaler_seam_on_outlier_features():
    """RobustScaler ↔ 팻테일 피처 seam: 폴드 train fit 후 val 유한·본체 보존."""
    df = _outlier_ohlcv(n=400, seed=0)
    X = build_features(df)
    cols = ["semi_dev", "kf_uncertainty", "kf_slope"]   # 진단상 왜곡 큰 축
    train_idx = X.index[130:190]
    val_idx = X.index[195:245]
    r = RobustScaler().fit(X.loc[train_idx, cols])
    z_val = r.transform(X.loc[val_idx, cols])
    assert np.isfinite(z_val.to_numpy()).all()
    # 팻테일 축도 본체 정보 보존(상수 압축 아님)
    for c in cols:
        assert z_val[c].std() > 0.05


# ---- 2.4-f: 실 1h full-stack smoke (skip-guard, 서브셋 — hurst 성능 F-13) ----

@requires_1h
def test_full_stack_smoke_real_1h():
    """실 1h 피처 분포가 두 정규화·splitter·baseline 을 관통(배관 실증).

    서브셋 6000봉 사용: hurst rolling.apply 가 전체 56k봉에 ~68s(F-13) — smoke 는 가볍게.
    라벨 NaN 드롭(F-10)은 clean y 로 분리(X seam 격리).
    """
    df = load_audited(SYM, "1h").iloc[:6000]
    X = build_features(df)
    assert X.shape[1] == 8

    # 실데이터 X ↔ 삼중배리어 라벨 index 계약
    bl = compute_triple_barrier(df, LabelParams("atr", 96, 3.0, 24))
    assert X.index.equals(bl.labels.index)

    y = _clean_labels(df.index)
    sp = WalkForwardSplitter(train_min=2000, val_size=1000, label_horizon=24, step=1000)
    for nf in (ZScoreNormalizer, RobustScaler):
        res = run_walk_forward(X, y, PriorBaseline, sp, normalizer_factory=nf, seed=0)
        assert res.n_folds >= 2
        assert res.report.per_fold.shape[0] == res.n_folds

    # 실 X 가 폴드 val 구간(워밍업 이후)서 유한
    for fold in sp.split(X.index):
        assert np.isfinite(X.loc[fold.val_index].to_numpy()).all()
