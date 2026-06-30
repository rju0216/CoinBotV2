"""strategy/helpers/ opt-in 공식 단위 테스트 (sizing / risk_gates)."""

from __future__ import annotations

import pytest

from src.core.types import AccountState
from src.strategy.helpers.risk_gates import (
    daily_loss_exceeded,
    drawdown_exceeded,
)
from src.strategy.helpers.sizing import risk_based_size


# ---- sizing ----

def test_risk_based_size_basic():
    # risk_amount = 10000 * 0.01 = 100, price_risk = |67000-66500| = 500
    # raw = 100/500 = 0.2 ; leverage cap = (10000*5)/67000 ≈ 0.746
    size = risk_based_size(
        67000.0, 66500.0, 10000.0,
        risk_per_trade_pct=0.01, max_leverage=5,
    )
    assert size == pytest.approx(0.2, rel=1e-6)


def test_risk_based_size_leverage_cap():
    # 좁은 SL → raw 큼 → leverage cap 적용
    size = risk_based_size(
        67000.0, 66990.0, 10000.0,
        risk_per_trade_pct=0.5, max_leverage=2,
    )
    assert size == pytest.approx((10000 * 2) / 67000, rel=1e-6)


def test_risk_based_size_max_position_clamp():
    size = risk_based_size(
        67000.0, 66500.0, 10000.0,
        risk_per_trade_pct=0.01, max_leverage=5, max_position_size=0.05,
    )
    assert size == 0.05


def test_risk_based_size_volatility_reduces():
    base = risk_based_size(
        67000.0, 66500.0, 10000.0,
        risk_per_trade_pct=0.01, max_leverage=5,
    )
    reduced = risk_based_size(
        67000.0, 66500.0, 10000.0,
        risk_per_trade_pct=0.01, max_leverage=5, volatility_factor=2.0,
    )
    assert reduced == pytest.approx(base * 0.5, rel=1e-6)


def test_risk_based_size_zero_distance():
    assert risk_based_size(
        67000.0, 67000.0, 10000.0,
        risk_per_trade_pct=0.01, max_leverage=5,
    ) == 0.0


# ---- risk_gates ----

def _account(balance=1000.0, equity=1000.0, peak=1000.0, daily=0.0, dd=0.0):
    return AccountState(
        balance=balance, equity=equity, peak_equity=peak,
        daily_pnl=daily, initial_balance=balance, drawdown_pct=dd,
    )


def test_drawdown_exceeded():
    assert drawdown_exceeded(_account(dd=0.36), 0.35) is True
    assert drawdown_exceeded(_account(dd=0.30), 0.35) is False


def test_daily_loss_exceeded():
    assert daily_loss_exceeded(_account(balance=1000.0, daily=-60.0), 0.05) is True
    assert daily_loss_exceeded(_account(balance=1000.0, daily=-40.0), 0.05) is False
    # 잔액 0 이하면 False (분모 보호)
    assert daily_loss_exceeded(_account(balance=0.0, daily=-100.0), 0.05) is False
