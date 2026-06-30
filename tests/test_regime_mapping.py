"""상태→Contract 매핑 단위 테스트 (D4-2c). 순수 로직 (HMM 적합 없음)."""

from __future__ import annotations

import numpy as np

from src.strategy.regime.contract import RegimeDirection, RegimeType
from src.strategy.regime.mapping import StateMapping, StateStats


def _dummy_stats(n):
    return tuple(StateStats(0.0, 1.0) for _ in range(n))


# ---- type/direction 분류 ----

def test_mapping_trend_long_and_range():
    T = 300
    rng = np.random.default_rng(0)
    r0 = rng.normal(0.01, 0.002, T)   # 강한 +추세 |μ|/std ≈ 5
    r1 = rng.normal(0.0, 0.01, T)     # range |μ|/std ≈ 0
    raw = np.concatenate([r0, r1])
    gamma = np.zeros((2 * T, 2))
    gamma[:T, 0] = 1.0
    gamma[T:, 1] = 1.0
    m = StateMapping.fit(raw, gamma, tau=1.0)
    assert m.types[0] == RegimeType.TREND
    assert m.directions[0] == RegimeDirection.LONG
    assert m.types[1] == RegimeType.RANGE
    assert m.directions[1] == RegimeDirection.NONE


def test_mapping_trend_short():
    T = 300
    rng = np.random.default_rng(1)
    r = rng.normal(-0.01, 0.002, T)   # 강한 -추세
    gamma = np.ones((T, 1))
    m = StateMapping.fit(r, gamma, tau=1.0)
    assert m.types[0] == RegimeType.TREND
    assert m.directions[0] == RegimeDirection.SHORT


def test_low_ratio_is_range_even_if_directional_noise():
    # mean 작고 std 큼 → |μ|/std < τ → range (방향 노이즈 무시)
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.02, 500)  # |μ|/std ≈ 0.05
    m = StateMapping.fit(r, np.ones((500, 1)), tau=1.0)
    assert m.types[0] == RegimeType.RANGE
    assert m.directions[0] == RegimeDirection.NONE


# ---- confidence 집계 (추론) ----

def test_aggregate_sums_confidence_over_same_contract():
    m = StateMapping(
        types=(RegimeType.TREND, RegimeType.TREND, RegimeType.RANGE),
        directions=(RegimeDirection.LONG, RegimeDirection.LONG, RegimeDirection.NONE),
        tau=1.0,
        stats=_dummy_stats(3),
    )
    c = m.aggregate(np.array([0.3, 0.4, 0.3]), volatility=500.0)
    assert c.type == RegimeType.TREND
    assert c.direction == RegimeDirection.LONG
    assert abs(c.confidence - 0.7) < 1e-12   # 0.3 + 0.4 집계
    assert c.volatility == 500.0


def test_aggregate_picks_max_mass_contract():
    m = StateMapping(
        types=(RegimeType.TREND, RegimeType.RANGE),
        directions=(RegimeDirection.LONG, RegimeDirection.NONE),
        tau=1.0,
        stats=_dummy_stats(2),
    )
    c = m.aggregate(np.array([0.3, 0.7]), volatility=100.0)  # range 가 더 큼
    assert c.type == RegimeType.RANGE
    assert abs(c.confidence - 0.7) < 1e-12


def test_covered_combos():
    m = StateMapping(
        types=(RegimeType.TREND, RegimeType.TREND, RegimeType.RANGE),
        directions=(RegimeDirection.LONG, RegimeDirection.SHORT, RegimeDirection.NONE),
        tau=1.0,
        stats=_dummy_stats(3),
    )
    combos = m.covered_combos()
    assert (RegimeType.TREND, RegimeDirection.LONG) in combos
    assert (RegimeType.TREND, RegimeDirection.SHORT) in combos
    assert (RegimeType.RANGE, RegimeDirection.NONE) in combos
    assert len(combos) == 3
