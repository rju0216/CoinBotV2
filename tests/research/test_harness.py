"""harness 회귀 테스트 — walk-forward 실행·정규화 hook·재현성."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.splitter import WalkForwardSplitter


def _xy(n):
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(
        {"f1": np.arange(n, dtype=float), "f2": (np.arange(n) % 5).astype(float)},
        index=idx,
    )
    y = pd.Series(np.array(["up", "down", "flat"])[np.arange(n) % 3], index=idx)
    return X, y


def test_run_walk_forward_structure():
    X, y = _xy(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, PriorBaseline, sp)
    assert res.n_folds >= 2
    assert len(res.report.per_fold) == res.n_folds
    assert {"balanced_accuracy", "mcc", "log_loss"} <= set(res.report.per_fold.columns)
    assert res.config["label_horizon"] == 10


def test_run_with_normalizer_hook():
    X, y = _xy(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, PriorBaseline, sp, normalizer_factory=ZScoreNormalizer)
    assert res.config["normalized"] is True
    assert res.n_folds >= 2


def test_labels_inferred_when_omitted():
    X, y = _xy(800)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=5)
    res = run_walk_forward(X, y, PriorBaseline, sp)
    assert set(res.config["labels"]) == {"up", "down", "flat"}


def test_seed_reproducible():
    X, y = _xy(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    r1 = run_walk_forward(X, y, PriorBaseline, sp, seed=42)
    r2 = run_walk_forward(X, y, PriorBaseline, sp, seed=42)
    assert r1.report.per_fold.equals(r2.report.per_fold)


def test_raises_when_data_too_short():
    X, y = _xy(50)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    with pytest.raises(ValueError):
        run_walk_forward(X, y, PriorBaseline, sp)


# ---- F-10: NaN 위생 (실모델·정규화 seam) ----

class _NaNRejectModel(PriorBaseline):
    """fit 입력에 NaN 이 있으면 실패 — 실모델(MLP/트리)의 NaN 불내성을 모사.

    NaN 드롭이 없으면 워밍업/동시터치 NaN 이 fit 에 유입돼 여기서 터진다(F-10 재현).
    predict_proba 는 val X 를 실제로 참조(NaN 이면 exception)해 val 측 드롭도 강제 검증.
    """

    def fit(self, X, y):
        if pd.DataFrame(X).isna().any().any():
            raise ValueError("fit 입력에 NaN (드롭 실패)")
        if pd.Series(y).isna().any():
            raise ValueError("y 에 NaN (드롭 실패)")
        return super().fit(X, y)

    def predict_proba(self, X):
        if pd.DataFrame(X).isna().any().any():
            raise ValueError("predict 입력에 NaN (val 드롭 실패)")
        return super().predict_proba(X)


class _NaNAssertNormalizer(ZScoreNormalizer):
    """fit 입력에 NaN 이 있으면 실패 — 드롭이 정규화 '이전'임을 강제(F-9 계약 보호)."""

    def fit(self, train_df):
        if pd.DataFrame(train_df).isna().any().any():
            raise ValueError("정규화 fit 이 NaN 을 받음 (드롭이 정규화 이후)")
        return super().fit(train_df)


def _xy_with_nan(n):
    """X 앞쪽에 워밍업 NaN(첫 20행) + y 중간에 동시터치 NaN(산발) 주입."""
    X, y = _xy(n)
    X = X.copy()
    y = y.copy()
    X.iloc[:20, 0] = np.nan          # 워밍업 NaN (한 열)
    y.iloc[350] = np.nan             # 중간 y NaN (train/val 걸침 위치)
    y.iloc[700] = np.nan
    return X, y


def test_nan_dropped_protects_model_fit():
    # NaN 드롭이 실모델 대용 stub 의 fit/predict 를 NaN 으로부터 보호한다.
    X, y = _xy_with_nan(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, _NaNRejectModel, sp)   # 드롭 없으면 ValueError
    assert res.n_folds >= 2


def test_drop_happens_before_normalizer():
    X, y = _xy_with_nan(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    # 정규화가 NaN 을 받으면 assert 실패 → 드롭이 정규화 이전임을 강제
    res = run_walk_forward(
        X, y, _NaNRejectModel, sp, normalizer_factory=_NaNAssertNormalizer
    )
    assert res.n_folds >= 2


def test_coverage_reports_dropped_rows():
    X, y = _xy_with_nan(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, PriorBaseline, sp)
    cov = res.coverage
    assert {"train_n", "train_dropped", "val_n", "val_dropped"} <= set(cov.columns)
    # 주입한 워밍업 20 + y NaN 2 개가 전 폴드 합에 반영 (train·val 어딘가에 귀속)
    total_dropped = int(cov["train_dropped"].sum() + cov["val_dropped"].sum())
    assert total_dropped >= 1
    assert res.config["val_dropped_total"] == int(cov["val_dropped"].sum())


def test_val_nan_excluded_from_scoring():
    # val 에 떨어진 y NaN 은 채점 표본에서 제외 (FoldPrediction.y_true 무 NaN)
    X, y = _xy_with_nan(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, PriorBaseline, sp)
    for fp in res.fold_preds:
        assert not fp.y_true.isna().any()
        assert fp.y_pred.index.equals(fp.y_true.index)


def test_clean_data_drops_nothing():
    # NaN 이 없으면 드롭 0 (비파괴) — 기존 동작 보존
    X, y = _xy(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    res = run_walk_forward(X, y, PriorBaseline, sp)
    assert int(res.coverage["train_dropped"].sum()) == 0
    assert int(res.coverage["val_dropped"].sum()) == 0
    assert res.config["val_dropped_total"] == 0
