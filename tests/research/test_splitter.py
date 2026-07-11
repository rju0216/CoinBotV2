"""walk-forward splitter 회귀 테스트 — 폴드 경계·purge 누수·꼬리예약·비겹침."""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.validation.splitter import Fold, WalkForwardSplitter


def _idx(n, freq="h"):
    return pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")


def _pos(index, ts):
    return index.get_loc(ts)


def test_basic_expanding_structure():
    idx = _idx(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    folds = list(sp.split(idx))
    assert len(folds) >= 2
    # 첫 폴드 학습 크기 = train_min (purge 후)
    assert folds[0].n_train == 300
    # expanding: 학습창이 폴드마다 자람
    for a, b in zip(folds, folds[1:]):
        assert b.n_train > a.n_train
    # 검증창 크기 일정
    assert all(f.n_val == 100 for f in folds)


def test_purge_prevents_label_overlap():
    idx = _idx(1000)
    N = 10
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=N)
    for f in sp.split(idx):
        last_train_pos = _pos(idx, f.train_index[-1])
        first_val_pos = _pos(idx, f.val_index[0])
        # 마지막 학습표본의 라벨창 [s, s+N] 이 val 시작에 닿지 않아야 함
        assert last_train_pos + N < first_val_pos
        assert f.n_purged == N          # embargo=0 → gap=N


def test_embargo_widens_gap():
    idx = _idx(1000)
    N, emb = 10, 5
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=N, embargo=emb)
    f = next(iter(sp.split(idx)))
    last_train_pos = _pos(idx, f.train_index[-1])
    first_val_pos = _pos(idx, f.val_index[0])
    assert first_val_pos - last_train_pos - 1 == N + emb
    assert f.n_purged == N + emb


def test_tail_reserved_for_unresolved_labels():
    idx = _idx(1000)
    N = 10
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=N)
    folds = list(sp.split(idx))
    last_val_pos = _pos(idx, folds[-1].val_index[-1])
    # 마지막 검증 봉의 라벨이 해소되려면 위치 <= n-1-N
    assert last_val_pos <= len(idx) - 1 - N


def test_validation_windows_non_overlapping():
    idx = _idx(1000)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    folds = list(sp.split(idx))
    for a, b in zip(folds, folds[1:]):
        assert a.val_index[-1] < b.val_index[0]


def test_train_and_val_disjoint():
    idx = _idx(800)
    sp = WalkForwardSplitter(train_min=200, val_size=80, label_horizon=5)
    for f in sp.split(idx):
        assert f.train_index.intersection(f.val_index).empty
        assert f.train_index[-1] < f.val_index[0]


def test_raise_when_too_short():
    idx = _idx(100)
    sp = WalkForwardSplitter(train_min=300, val_size=100, label_horizon=10)
    with pytest.raises(ValueError):
        list(sp.split(idx))


def test_requires_monotonic_index():
    idx = _idx(500)[::-1]  # 역순
    sp = WalkForwardSplitter(train_min=100, val_size=50, label_horizon=5)
    with pytest.raises(ValueError):
        list(sp.split(idx))


def test_invalid_params():
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_min=0, val_size=100, label_horizon=10)
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_min=100, val_size=100, label_horizon=-1)


def test_step_controls_fold_count():
    idx = _idx(2000)
    # step = val_size (비겹침)
    n1 = len(list(WalkForwardSplitter(300, 100, 10).split(idx)))
    # step 절반 → 폴드 수 대략 2배 (검증창 겹침)
    n2 = len(list(WalkForwardSplitter(300, 100, 10, step=50).split(idx)))
    assert n2 > n1


def test_fold_is_dataclass_with_spans():
    idx = _idx(600)
    f = next(iter(WalkForwardSplitter(200, 80, 5).split(idx)))
    assert isinstance(f, Fold)
    assert f.train_span[0] == idx[0]
    assert f.val_span[0] == f.val_index[0]
