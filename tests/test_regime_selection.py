"""K 선택 단위 테스트 (D4-2c).

recommend_k: 순수 로직 (적합 없음, 빠름).
evaluate_k: 1개 통합 테스트 (3-레짐 합성 → 커버리지 진단).
"""

from __future__ import annotations

import numpy as np

from src.strategy.regime.contract import RegimeDirection, RegimeType
from src.strategy.regime.selection import (
    CORE_COMBOS,
    KDiagnostic,
    _core_combos,
    evaluate_k,
    recommend_k,
)

_T, _L, _S = RegimeType.TREND, RegimeDirection.LONG, RegimeDirection.SHORT
_R, _N = RegimeType.RANGE, RegimeDirection.NONE


def _diag(k, ll, bic, covered, core):
    return KDiagnostic(k, ll, bic, frozenset(covered), core)


# ---- recommend_k 순수 로직 ----

def test_recommend_min_bic_among_coverage():
    diags = [
        _diag(2, -100.0, 250.0, {(_R, _N), (_T, _L)}, False),
        _diag(3, -90.0, 240.0, set(CORE_COMBOS), True),
        _diag(4, -88.0, 245.0, set(CORE_COMBOS), True),
    ]
    assert recommend_k(diags) == 3  # 커버리지 통과(3,4) 중 BIC 최소


def test_recommend_fallback_to_holdout_when_no_coverage():
    diags = [
        _diag(2, -100.0, 250.0, {(_R, _N)}, False),
        _diag(3, -85.0, 240.0, {(_R, _N), (_T, _L)}, False),
    ]
    assert recommend_k(diags) == 3  # 코어 커버리지 없음 → holdout LL 최대


# ---- evaluate_k 통합 (3-레짐 합성) ----

def test_evaluate_k_integration_core_coverage():
    rng = np.random.default_rng(0)
    K, d = 3, 3
    feat_means = np.array([[2, 2, 0], [-2, 2, 0], [0, -2, 0]], float)  # long/short/range
    ret_mean = np.array([0.01, -0.01, 0.0])    # |μ|/std ≈ 5, 5, 0
    ret_std = np.array([0.002, 0.002, 0.01])
    A = np.array(
        [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
    )
    n = 2400
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.array([rng.multivariate_normal(feat_means[s], np.eye(d)) for s in states])
    raw_ret = np.array([rng.normal(ret_mean[s], ret_std[s]) for s in states])

    diag, hmm, mapping = evaluate_k(
        X[:1800], raw_ret[:1800], X[1800:], k=3, tau=1.0, n_init=4, seed=0,
    )
    assert diag.k == 3
    assert isinstance(diag.bic, float)
    assert isinstance(diag.holdout_ll, float)
    # 잘 분리된 3-레짐 → 코어 커버리지(trend/long·trend/short·range/none) 충족
    assert diag.has_core_coverage
    assert CORE_COMBOS.issubset(mapping.covered_combos())


# ---- D4-5 S2: emission 배선(I-013) + core_combos 동적(C-1) ----

def _synth_3regime(seed):
    """long/short/range 3-레짐 합성 (X feature + raw 로그수익률)."""
    rng = np.random.default_rng(seed)
    K, d = 3, 3
    feat_means = np.array([[2, 2, 0], [-2, 2, 0], [0, -2, 0]], float)
    ret_mean = np.array([0.01, -0.01, 0.0])   # |μ|/std ≈ 5, 5, 0
    ret_std = np.array([0.002, 0.002, 0.01])
    A = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])
    n = 2400
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.array([rng.multivariate_normal(feat_means[s], np.eye(d)) for s in states])
    raw_ret = np.array([rng.normal(ret_mean[s], ret_std[s]) for s in states])
    return X, raw_ret


def test_core_combos_trend_only_excludes_range():
    trend_only = _core_combos(False)
    assert trend_only == frozenset({(_T, _L), (_T, _S)})
    assert (_R, _N) not in trend_only
    assert _core_combos(True) == CORE_COMBOS  # range 활성 = 하위호환


def test_evaluate_k_student_t_wired():
    # student_t emission K선택 배선(I-013) + BIC 에 ν 반영(상태별 ν → K 만큼 파라미터↑).
    X, raw_ret = _synth_3regime(0)
    diag_g, hmm_g, _ = evaluate_k(
        X[:1800], raw_ret[:1800], X[1800:], k=3, tau=1.0, n_init=2, seed=0,
        emission_kind="gaussian",
    )
    diag_t, hmm_t, _ = evaluate_k(
        X[:1800], raw_ret[:1800], X[1800:], k=3, tau=1.0, n_init=2, seed=0,
        emission_kind="student_t",
    )
    assert np.isfinite(diag_t.bic) and np.isfinite(diag_t.holdout_ll)
    assert hmm_t.emission.kind == "student_t"
    # t 는 상태별 ν(K개) 추가 → n_params 가 Gaussian 보다 정확히 K 만큼 큼 (BIC 반영)
    assert hmm_t.n_params == hmm_g.n_params + 3


def test_evaluate_k_enable_range_false_core():
    # enable_range=False → 비trend 는 NONE, core 커버리지 = {trend L/S} (range 제외).
    X, raw_ret = _synth_3regime(1)
    diag, _, mapping = evaluate_k(
        X[:1800], raw_ret[:1800], X[1800:], k=3, tau=1.0, n_init=4, seed=0,
        enable_range=False,
    )
    covered = mapping.covered_combos()
    assert (_R, _N) not in covered          # range 상태 → NONE 이므로 RANGE 미출현
    assert diag.has_core_coverage           # 추세단독 core = {trend L/S} 충족
    assert _core_combos(False).issubset(covered)
