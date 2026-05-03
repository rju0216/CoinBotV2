"""확장 메트릭 helper (Phase E-2-4 Step 2).

evaluate_models 결과 분석용. Sharpe / Calmar / Bootstrap p-value를 계산하여
30 specs 매트릭스 비교 + 모델 간 통계 유의성 검정 지원.

기본 metrics(`BacktestEngine._build_metrics`의 total_return_pct/MDD/PF/win_rate)는
이미 백테 시 자동 계산. 본 모듈은 그 결과 위에 추가 분석을 얹는다.

미니 사안 결정 (Phase E-2-4):
- annualization = 365 (crypto 24/7 표준, merge_yearly_reports.py와 일관)
- Bootstrap n=10,000, seed=42 (재현성)
- 무위험 수익률 = 0 (BTC 단순화)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 365  # crypto 24/7 거래
DEFAULT_BOOTSTRAP_N = 10_000
DEFAULT_SEED = 42


def compute_sharpe_ratio(
    equity_curve: pd.DataFrame,
    annualization_days: int = ANNUALIZATION_DAYS,
) -> float:
    """equity_curve(timestamp index, balance/equity 컬럼) → annualized Sharpe.

    daily resample → pct_change → mean/std × √annualization. 무위험 수익률 0 가정.

    Returns:
        Sharpe ratio. 데이터 부족 또는 std=0 시 0.0.
    """
    if equity_curve is None or equity_curve.empty:
        return 0.0
    if "equity" in equity_curve.columns:
        eq = equity_curve["equity"]
    elif "balance" in equity_curve.columns:
        eq = equity_curve["balance"]
    else:
        return 0.0

    daily = eq.resample("1D").last().dropna()
    if len(daily) < 2:
        return 0.0
    returns = daily.pct_change().dropna()
    if len(returns) < 1 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(annualization_days))


def compute_calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
    oos_years: float,
) -> float:
    """연환산 수익률 / 최대 낙폭.

    Args:
        total_return_pct: 백테 기간 전체 수익률 (예: 5088.20)
        max_drawdown_pct: 최대 낙폭 (예: 6.28)
        oos_years: OOS 기간 (예: 1.0, 4.0)

    Returns:
        Calmar ratio. max_drawdown_pct=0 시 0.0 (∞ 회피).
    """
    if max_drawdown_pct <= 0 or oos_years <= 0:
        return 0.0
    # 연환산 수익률 = (1 + total_return/100) ** (1/years) - 1
    annual_return = (1 + total_return_pct / 100.0) ** (1.0 / oos_years) - 1
    return float((annual_return * 100.0) / max_drawdown_pct)


def bootstrap_pnl_diff(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    n: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float, float]:
    """두 모델 거래 PnL 분포 차이 검정 (pooled bootstrap).

    Null hypothesis: pnl_a와 pnl_b가 같은 분포에서 추출.
    pooled에서 n번 random resample하여 두 그룹 평균 차이 분포 형성 →
    실제 관찰된 차이의 |값|이 분포에서 어디 위치하는지 p-value.

    Args:
        pnl_a, pnl_b: 두 모델의 거래별 pnl (numpy array)
        n: bootstrap 반복 횟수
        seed: 재현성

    Returns:
        (observed_diff, p_value, ci_low, ci_high) — observed_diff는
        pnl_a.mean - pnl_b.mean. ci는 bootstrap 분포의 95% CI.
    """
    pnl_a = np.asarray(pnl_a, dtype=np.float64)
    pnl_b = np.asarray(pnl_b, dtype=np.float64)
    if len(pnl_a) == 0 or len(pnl_b) == 0:
        return 0.0, 1.0, 0.0, 0.0

    observed = float(pnl_a.mean() - pnl_b.mean())
    pooled = np.concatenate([pnl_a, pnl_b])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    n_a = len(pnl_a)
    total = len(pooled)
    for i in range(n):
        sample = rng.choice(pooled, size=total, replace=True)
        diffs[i] = sample[:n_a].mean() - sample[n_a:].mean()

    # two-tailed p-value
    p_value = float(np.mean(np.abs(diffs) >= np.abs(observed)))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return observed, p_value, ci_low, ci_high


def split_to_oos_years(split_id: str) -> float:
    """split_id → OOS 길이 (년). build_specs SPLIT_DEFINITIONS와 일관."""
    return {
        "1": 1.0, "A": 1.0, "B": 1.0,
        "Exp2": 2.0, "Exp3": 3.0, "Exp4": 4.0,
    }.get(split_id, 1.0)
