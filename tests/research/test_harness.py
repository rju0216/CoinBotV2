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
