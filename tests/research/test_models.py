"""Phase 3 모델 콜러블 회귀 — TreeBench·SmallMLP.

검증: ① ProbaModel 계약(형태·classes_·행합1·predict argmax) ② 재현성(seed) ③ 학습성
(신호 있으면 uniform 초과, 없으면 근처) ④ 단일클래스 폴드 폴백 ⑤ harness seam
(모델 ↔ NaN 드롭·정규화·폴드, 규칙 16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.models import SmallMLP, TreeBench
from src.research.normalize import ZScoreNormalizer
from src.research.validation.harness import run_walk_forward
from src.research.validation.metrics import balanced_accuracy
from src.research.validation.splitter import WalkForwardSplitter

CLASSES = ["down", "expire", "up"]
_FAST = {"max_epochs": 40}   # MLP 테스트 속도용 (placeholder 축소)


def _signal_xy(n, seed=0, noise=0.0):
    """x0 부호가 라벨을 정하는 학습 가능 신호. noise 로 난이도 조절."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    x0 = rng.normal(size=n)
    other = rng.normal(size=(n, 3))
    X = pd.DataFrame(
        np.column_stack([x0, other]), index=idx, columns=["f0", "f1", "f2", "f3"]
    )
    eff = x0 + noise * rng.normal(size=n)
    y = np.where(eff > 0.5, "up", np.where(eff < -0.5, "down", "expire"))
    return X, pd.Series(y, index=idx)


def _no_signal_xy(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=["f0", "f1", "f2", "f3"])
    y = pd.Series(rng.choice(CLASSES, size=n), index=idx)
    return X, y


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=0),
    lambda: SmallMLP(seed=0, params=_FAST),
])
def test_contract_shape(factory):
    X, y = _signal_xy(400)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    m = factory().fit(Xtr, ytr)
    assert m.classes_ == sorted(set(ytr))
    proba = m.predict_proba(Xte)
    assert list(proba.columns) == m.classes_
    assert proba.index.equals(Xte.index)
    assert np.allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-5)
    pred = m.predict(Xte)
    assert len(pred) == len(Xte)
    assert set(np.unique(pred)) <= set(m.classes_)


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=7),
    lambda: SmallMLP(seed=7, params=_FAST),
])
def test_reproducible_same_seed(factory):
    X, y = _signal_xy(400)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    p1 = factory().fit(Xtr, ytr).predict_proba(Xte)
    p2 = factory().fit(Xtr, ytr).predict_proba(Xte)
    pd.testing.assert_frame_equal(p1, p2)


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=0),
    lambda: SmallMLP(seed=0, params=_FAST),
])
def test_learns_signal(factory):
    # 명확한 신호 → 홀드아웃 균형정확도가 uniform(1/3) 명백 초과
    X, y = _signal_xy(1200, noise=0.1)
    Xtr, ytr = X.iloc[:900], y.iloc[:900]
    Xte, yte = X.iloc[900:], y.iloc[900:]
    m = factory().fit(Xtr, ytr)
    ba = balanced_accuracy(yte, m.predict(Xte))
    assert ba > 0.55


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=0),
    lambda: SmallMLP(seed=0, params=_FAST),
])
def test_no_signal_near_chance(factory):
    # 신호 없음 → 홀드아웃 균형정확도가 우연 수준 근처(과신 안 함)
    X, y = _no_signal_xy(1200)
    Xtr, ytr = X.iloc[:900], y.iloc[:900]
    Xte, yte = X.iloc[900:], y.iloc[900:]
    m = factory().fit(Xtr, ytr)
    ba = balanced_accuracy(yte, m.predict(Xte))
    assert ba < 0.45


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=0),
    lambda: SmallMLP(seed=0, params=_FAST),
])
def test_single_class_fold_fallback(factory):
    # 단일클래스 train → 상수확률 폴백(크래시 없음)
    idx = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 4)), index=idx)
    y = pd.Series(["up"] * 100, index=idx)
    m = factory().fit(X, y)
    proba = m.predict_proba(X.iloc[:10])
    assert m.classes_ == ["up"]
    assert np.allclose(proba.to_numpy(), 1.0)


def test_mlp_seed_varies_output():
    # 다른 seed → 다른 초기화(다중 seed 안정성 측정의 근거, 보강1)
    X, y = _signal_xy(400)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    p0 = SmallMLP(seed=0, params=_FAST).fit(Xtr, ytr).predict_proba(Xte)
    p1 = SmallMLP(seed=1, params=_FAST).fit(Xtr, ytr).predict_proba(Xte)
    assert not np.allclose(p0.to_numpy(), p1.to_numpy())


@pytest.mark.parametrize("factory", [
    lambda: TreeBench(seed=0),
    lambda: SmallMLP(seed=0, params=_FAST),
])
def test_harness_seam_with_nan_and_normalize(factory):
    # 모델 ↔ NaN 드롭(Step3.1) ↔ 정규화 ↔ 폴드 통합 (규칙 16)
    X, y = _signal_xy(1000)
    X = X.copy()
    X.iloc[:20, 0] = np.nan             # 워밍업 NaN
    y = y.copy()
    y.iloc[400] = np.nan                # 중간 y NaN
    sp = WalkForwardSplitter(train_min=300, val_size=150, label_horizon=10)
    res = run_walk_forward(
        X, y, factory, sp, labels=CLASSES,
        normalizer_factory=ZScoreNormalizer, seed=0,
    )
    assert res.n_folds >= 2
    assert {"balanced_accuracy", "mcc", "log_loss"} <= set(res.report.per_fold.columns)
    assert int(res.coverage["train_dropped"].sum() + res.coverage["val_dropped"].sum()) >= 1


# ---- R3.0: SmallMLP 멀티태스크 (도달시간 co-training) ----

def _reach_for(y):
    """도달시간 대용 회귀 타깃: up/down 은 [0.1,1.0], expire 는 NaN(검열 제외 마스킹)."""
    rng = np.random.default_rng(1)
    r = rng.uniform(0.1, 1.0, len(y))
    r[y.to_numpy() == "expire"] = np.nan
    return pd.Series(r, index=y.index)


def test_mlp_multitask_contract():
    # reach 제공 → 멀티태스크 fit, predict_proba 는 여전히 배리어 계약 준수
    X, y = _signal_xy(400)
    reach = _reach_for(y)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    m = SmallMLP(seed=0, params=_FAST, reach=reach).fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    assert list(proba.columns) == m.classes_
    assert proba.index.equals(Xte.index)
    assert np.allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-5)


def test_mlp_multitask_reproducible():
    X, y = _signal_xy(400)
    reach = _reach_for(y)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    p1 = SmallMLP(seed=0, params=_FAST, reach=reach).fit(Xtr, ytr).predict_proba(Xte)
    p2 = SmallMLP(seed=0, params=_FAST, reach=reach).fit(Xtr, ytr).predict_proba(Xte)
    pd.testing.assert_frame_equal(p1, p2)


def test_mlp_multitask_differs_from_single_task():
    # 멀티태스크(reach head + co-training)는 단일태스크와 다른 예측 → 헤드가 실제 작동
    X, y = _signal_xy(400)
    reach = _reach_for(y)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    single = SmallMLP(seed=0, params=_FAST).fit(Xtr, ytr).predict_proba(Xte)
    multi = SmallMLP(seed=0, params=_FAST, reach=reach).fit(Xtr, ytr).predict_proba(Xte)
    assert not np.allclose(single.to_numpy(), multi.to_numpy())


def test_mlp_single_task_unchanged_by_reach_none():
    # reach=None(기본) = 단일태스크 = R1 경로 (reach 헤드 미생성, 초기화 RNG 불변)
    X, y = _signal_xy(400)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    a = SmallMLP(seed=0, params=_FAST).fit(Xtr, ytr).predict_proba(Xte)
    b = SmallMLP(seed=0, params=_FAST, reach=None).fit(Xtr, ytr).predict_proba(Xte)
    pd.testing.assert_frame_equal(a, b)


def test_mlp_reach_all_nan_no_crash():
    # reach 전량 NaN(검열 제외) → reach 손실 0 가드, 크래시 없음
    X, y = _signal_xy(400)
    reach = pd.Series(np.nan, index=y.index)
    Xtr, ytr, Xte = X.iloc[:300], y.iloc[:300], X.iloc[300:]
    m = SmallMLP(seed=0, params=_FAST, reach=reach).fit(Xtr, ytr)
    assert m.predict_proba(Xte).shape[0] == len(Xte)
