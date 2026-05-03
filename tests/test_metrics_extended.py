"""metrics_extended 단위 테스트 (Phase E-2-4 Step 2).

회귀 보호: Sharpe / Calmar / Bootstrap 알고리즘 정확성 보증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.metrics_extended import (
    bootstrap_pnl_diff,
    compute_calmar_ratio,
    compute_sharpe_ratio,
    split_to_oos_years,
)


def _make_equity(values: list[float], start: str = "2025-01-01", freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    df = pd.DataFrame({"balance": values, "equity": values}, index=idx)
    df.index.name = "timestamp"
    return df


class TestSharpeRatio:
    def test_empty_returns_zero(self):
        df = pd.DataFrame(columns=["balance", "equity"])
        assert compute_sharpe_ratio(df) == 0.0

    def test_constant_balance_returns_zero(self):
        # 변동 없는 잔고 → returns std=0 → Sharpe 0
        df = _make_equity([10000.0] * 200)
        assert compute_sharpe_ratio(df) == 0.0

    def test_linear_growth_positive_sharpe(self):
        # 단조 증가 → 양수 Sharpe
        n = 30 * 96  # 30일 × 96 (15분 × 4 × 24)
        values = [10000.0 * (1.0 + 0.001 * i) for i in range(n)]
        df = _make_equity(values)
        sharpe = compute_sharpe_ratio(df)
        assert sharpe > 0

    def test_volatile_lower_sharpe(self):
        # 같은 평균 수익이지만 변동성 큰 케이스 → Sharpe 낮음
        n = 30 * 96
        rng = np.random.default_rng(0)
        values_stable = [10000.0 * (1.0 + 0.001 * i) for i in range(n)]
        # 노이즈 추가
        noise = rng.normal(0, 200, n)
        values_volatile = [v + noise[i] for i, v in enumerate(values_stable)]
        s_stable = compute_sharpe_ratio(_make_equity(values_stable))
        s_volatile = compute_sharpe_ratio(_make_equity(values_volatile))
        assert s_stable > s_volatile


class TestCalmarRatio:
    def test_basic_calculation(self):
        # 100% 수익 1년, MDD 10% → Calmar = 100/10 = 10
        cal = compute_calmar_ratio(100.0, 10.0, 1.0)
        assert abs(cal - 10.0) < 0.01

    def test_zero_drawdown_returns_zero(self):
        # MDD 0이면 0 반환 (∞ 회피)
        assert compute_calmar_ratio(100.0, 0.0, 1.0) == 0.0

    def test_negative_drawdown_returns_zero(self):
        assert compute_calmar_ratio(100.0, -5.0, 1.0) == 0.0

    def test_zero_oos_years_returns_zero(self):
        assert compute_calmar_ratio(100.0, 10.0, 0.0) == 0.0

    def test_4year_annualization(self):
        # 100% 수익 4년 → 연환산 ~18.92%, MDD 10% → Calmar ~1.892
        cal = compute_calmar_ratio(100.0, 10.0, 4.0)
        expected = ((1 + 1.0) ** 0.25 - 1) * 100 / 10
        assert abs(cal - expected) < 0.01


class TestBootstrap:
    def test_same_distribution_high_p_value(self):
        # 같은 분포 → null hypothesis 기각 못 함 → p > 0.05
        rng = np.random.default_rng(42)
        a = rng.normal(0, 100, 500)
        b = rng.normal(0, 100, 500)
        _, p, _, _ = bootstrap_pnl_diff(a, b, n=2000, seed=42)
        assert p > 0.05

    def test_clearly_different_distributions_low_p_value(self):
        # 명확히 다른 평균 → null hypothesis 기각 → p < 0.05
        rng = np.random.default_rng(42)
        a = rng.normal(100, 50, 500)
        b = rng.normal(-100, 50, 500)
        _, p, _, _ = bootstrap_pnl_diff(a, b, n=2000, seed=42)
        assert p < 0.05

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 200)
        b = rng.normal(20, 100, 200)
        d1, p1, _, _ = bootstrap_pnl_diff(a, b, n=1000, seed=42)
        d2, p2, _, _ = bootstrap_pnl_diff(a, b, n=1000, seed=42)
        assert d1 == d2
        assert p1 == p2

    def test_empty_inputs_safe(self):
        d, p, low, high = bootstrap_pnl_diff(np.array([]), np.array([1.0]))
        assert d == 0.0 and p == 1.0


class TestSplitToOosYears:
    @pytest.mark.parametrize("split,years", [
        ("1", 1.0), ("A", 1.0), ("B", 1.0),
        ("Exp2", 2.0), ("Exp3", 3.0), ("Exp4", 4.0),
        ("unknown", 1.0),
    ])
    def test_split_id_mapping(self, split, years):
        assert split_to_oos_years(split) == years
