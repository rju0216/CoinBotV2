"""TrendLogic 단위 테스트 (S1) — 추세 매매 정책 격리 검증."""

from __future__ import annotations

from datetime import datetime, timezone

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import Position
from src.strategy.regime.contract import (
    Contract,
    RegimeDirection,
    RegimeType,
)
from src.strategy.regime.trend_logic import TrendLogic


def _contract(type_, direction, conf, vol=100.0):
    return Contract(type=type_, direction=direction, confidence=conf, volatility=vol)


def _trend(direction, conf, vol=100.0):
    return _contract(RegimeType.TREND, direction, conf, vol)


def _range(conf=0.9, vol=100.0):
    return _contract(RegimeType.RANGE, RegimeDirection.NONE, conf, vol)


def _pos(side, entry=50000.0, sl=None):
    return Position(
        side=side,
        size=1.0,
        entry_price=entry,
        entry_time=datetime.now(timezone.utc),
        strategy_name="regime_quant",
        stop_loss=sl,
    )


PARAMS = {
    "theta_trend": 0.6,
    "k_trend_sl": 2.0,
    "k_trend_trail": 2.5,
    "risk_per_trade_pct": 0.01,
    "max_leverage": 5.0,
}


# ---- 진입 ----

def test_entry_long_above_theta():
    logic = TrendLogic(PARAMS)
    sig = logic.entry_signal(_trend(RegimeDirection.LONG, 0.7))
    assert sig is not None
    assert sig.side == SignalSide.LONG
    assert sig.confidence == 0.7
    assert sig.meta["volatility"] == 100.0


def test_entry_short_above_theta():
    logic = TrendLogic(PARAMS)
    sig = logic.entry_signal(_trend(RegimeDirection.SHORT, 0.95))
    assert sig is not None and sig.side == SignalSide.SHORT


def test_entry_blocked_below_theta():
    logic = TrendLogic(PARAMS)
    assert logic.entry_signal(_trend(RegimeDirection.LONG, 0.59)) is None


def test_entry_none_when_range():
    logic = TrendLogic(PARAMS)
    assert logic.entry_signal(_range(0.99)) is None


def test_entry_none_when_direction_none():
    logic = TrendLogic(PARAMS)
    # trend 인데 direction=none (방어적 — 정상 매핑에선 trend면 long/short)
    assert logic.entry_signal(_trend(RegimeDirection.NONE, 0.9)) is None


# ---- 초기 SL / 사이징 ----

def test_initial_stop_loss_long_short():
    logic = TrendLogic(PARAMS)
    long_sig = logic.entry_signal(_trend(RegimeDirection.LONG, 0.8))
    short_sig = logic.entry_signal(_trend(RegimeDirection.SHORT, 0.8))
    # k_trend_sl=2.0, vol=100 → 거리 200
    assert logic.initial_stop_loss(long_sig, 50000.0, 100.0) == 50000.0 - 200.0
    assert logic.initial_stop_loss(short_sig, 50000.0, 100.0) == 50000.0 + 200.0


def test_position_size_scales_with_confidence():
    logic = TrendLogic(PARAMS)
    entry, sl, balance = 50000.0, 49800.0, 10000.0
    sig_hi = logic.entry_signal(_trend(RegimeDirection.LONG, 1.0))
    sig_lo = logic.entry_signal(_trend(RegimeDirection.LONG, 0.6))
    size_hi = logic.position_size(sig_hi, entry, sl, balance)
    size_lo = logic.position_size(sig_lo, entry, sl, balance)
    assert size_hi > 0
    # confidence 0.6 은 1.0 의 0.6 배 (risk_based_size 동일 입력)
    assert abs(size_lo - size_hi * 0.6) < 1e-9


# ---- 트레일링 ----

def test_trailing_long_advances_only_favorably():
    logic = TrendLogic(PARAMS)
    c = _trend(RegimeDirection.LONG, 0.8, vol=100.0)
    # k_trend_trail=2.5, vol=100 → 거리 250. price=51000 → new_sl=50750
    pos = _pos(PositionSide.LONG, sl=50000.0)
    assert logic.trailing_stop(c, pos, 51000.0) == 50750.0
    # 가격이 내려가 new_sl 이 기존보다 낮으면 None(안 내림)
    pos2 = _pos(PositionSide.LONG, sl=50800.0)
    assert logic.trailing_stop(c, pos2, 51000.0) is None


def test_trailing_short_advances_only_favorably():
    logic = TrendLogic(PARAMS)
    c = _trend(RegimeDirection.SHORT, 0.8, vol=100.0)
    pos = _pos(PositionSide.SHORT, sl=50000.0)
    # price=49000 → new_sl=49250 < 50000 → 유리(SHORT 은 낮을수록 유리) → 갱신
    assert logic.trailing_stop(c, pos, 49000.0) == 49250.0
    pos2 = _pos(PositionSide.SHORT, sl=49200.0)
    assert logic.trailing_stop(c, pos2, 49000.0) is None


def test_trailing_freezes_on_trend_to_range():
    logic = TrendLogic(PARAMS)
    pos = _pos(PositionSide.LONG, sl=50000.0)
    assert logic.trailing_stop(_range(0.9), pos, 60000.0) is None


def test_trailing_freezes_on_adverse_trend():
    logic = TrendLogic(PARAMS)
    pos = _pos(PositionSide.LONG, sl=50000.0)
    # 반대방향 trend → 트레일링 동결(None), 청산은 force_exit 담당
    assert logic.trailing_stop(_trend(RegimeDirection.SHORT, 0.9), pos, 60000.0) is None


# ---- 강제 청산 ----

def test_force_exit_on_adverse_trend():
    logic = TrendLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    dec = logic.force_exit(_trend(RegimeDirection.SHORT, 0.8), pos)
    assert dec is not None and dec.reason == ExitReason.REGIME_EXIT


def test_no_force_exit_on_favorable_trend():
    logic = TrendLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    assert logic.force_exit(_trend(RegimeDirection.LONG, 0.8), pos) is None


def test_no_force_exit_on_range():
    logic = TrendLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    assert logic.force_exit(_range(0.9), pos) is None
