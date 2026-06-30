"""RangeLogic 단위 테스트 (S1) — 평균회귀 상태기계 + I-008 bounded 지표."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import Position
from src.strategy.regime.contract import (
    Contract,
    RegimeDirection,
    RegimeType,
)
from src.strategy.regime.range_logic import RangeLogic, _RangeState


def _contract(type_, direction, conf, vol=100.0):
    return Contract(type=type_, direction=direction, confidence=conf, volatility=vol)


def _range(conf=0.9, vol=100.0):
    return _contract(RegimeType.RANGE, RegimeDirection.NONE, conf, vol)


def _trend(direction, conf=0.9, vol=100.0):
    return _contract(RegimeType.TREND, direction, conf, vol)


def _pos(side, entry=50000.0, sl=None):
    return Position(
        side=side,
        size=1.0,
        entry_price=entry,
        entry_time=datetime.now(timezone.utc),
        strategy_name="regime_quant",
        stop_loss=sl,
    )


def _candles(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, dtype=float, index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c),
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": 1.0,
        },
        index=idx,
    )


PARAMS = {
    "theta_range": 0.6,
    "k_range_sl": 1.5,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "bb_period": 20,
    "bb_std": 2.0,
    "expiry_atr_mult": 1.0,
    "risk_per_trade_pct": 0.01,
    "max_leverage": 5.0,
    "indicator_window": 100,
}


def _stub_indicators(logic, lower, mid, upper, rsi):
    """_indicators 를 고정값으로 대체 — 상태기계 분기를 결정론적으로 검증."""
    logic._indicators = lambda candles: (lower, mid, upper, rsi)


# close 만 의미 있는 더미 캔들 (지표는 stub)
def _close_only(close):
    return _candles([close, close, close])


# ---- 셋업 (IDLE → ARMED) ----

def test_setup_long_arms():
    logic = RangeLogic(PARAMS)
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=25.0)
    # close < lower(100) AND rsi(25) <= oversold(30) → ARMED_LONG
    sig = logic.advance(_range(), _close_only(95.0))
    assert sig is None
    assert logic.state_name == "ARMED_LONG"


def test_setup_short_arms():
    logic = RangeLogic(PARAMS)
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=75.0)
    sig = logic.advance(_range(), _close_only(125.0))
    assert sig is None
    assert logic.state_name == "ARMED_SHORT"


def test_no_setup_when_only_band_breach():
    logic = RangeLogic(PARAMS)
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=45.0)
    # 밴드 이탈했지만 RSI 과매도 아님 → 셋업 안 됨 (AND 조건)
    logic.advance(_range(), _close_only(95.0))
    assert logic.state_name == "IDLE"


# ---- 트리거 (ARMED → 진입) ----

def test_trigger_long_on_return():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_LONG
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=35.0)
    # close(105) >= lower(100) → 밴드 안 복귀 → LONG 진입
    sig = logic.advance(_range(0.85), _close_only(105.0))
    assert sig is not None and sig.side == SignalSide.LONG
    assert sig.confidence == 0.85
    assert logic.state_name == "IDLE"  # 진입 후 reset


def test_trigger_short_on_return():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_SHORT
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=65.0)
    sig = logic.advance(_range(), _close_only(115.0))
    assert sig is not None and sig.side == SignalSide.SHORT
    assert logic.state_name == "IDLE"


# ---- 만료/리셋 (ARMED → IDLE 또는 중단) ----

def test_armed_long_distance_expiry():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_LONG
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=20.0)
    # close(50) < lower(100) - 1.0*atr(100)=0 → 사실 50 > 0 ... 거리 만료 조건 확인
    # lower - expiry*atr = 100 - 100 = 0. close=50 > 0 → 만료 아님, rsi 20<=30 유지 → 대기
    sig = logic.advance(_range(), _close_only(50.0))
    assert sig is None and logic.state_name == "ARMED_LONG"
    # close(-10) < 0 → 거리 만료 → IDLE
    sig2 = logic.advance(_range(), _close_only(-10.0))
    assert sig2 is None and logic.state_name == "IDLE"


def test_armed_long_rsi_relief_resets():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_LONG
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=40.0)
    # close(95) 여전히 밴드 밖(<100) 이지만 rsi(40) > oversold(30) → 모멘텀 소강 → IDLE
    sig = logic.advance(_range(), _close_only(95.0))
    assert sig is None and logic.state_name == "IDLE"


def test_armed_short_distance_expiry():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_SHORT
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=80.0)
    # upper + expiry*atr = 120 + 100 = 220. close=150 → 트리거(<=120)X·만료(>220)X·rsi(80<70)X → 대기
    sig = logic.advance(_range(), _close_only(150.0))
    assert sig is None and logic.state_name == "ARMED_SHORT"
    # close=300 > 220 → 거리 만료 → IDLE
    sig2 = logic.advance(_range(), _close_only(300.0))
    assert sig2 is None and logic.state_name == "IDLE"


def test_armed_short_rsi_relief_resets():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_SHORT
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=60.0)
    # close(125) 여전히 밴드 밖(>120) 이지만 rsi(60) < overbought(70) → 모멘텀 소강 → IDLE
    sig = logic.advance(_range(), _close_only(125.0))
    assert sig is None and logic.state_name == "IDLE"


def test_regime_exit_aborts_state_machine():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_LONG
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=20.0)
    # 레짐이 range 이탈(trend) → 즉시 중단(reset)
    sig = logic.advance(_trend(RegimeDirection.LONG), _close_only(95.0))
    assert sig is None and logic.state_name == "IDLE"


def test_low_confidence_aborts():
    logic = RangeLogic(PARAMS)
    logic._state = _RangeState.ARMED_LONG
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=20.0)
    # conf < theta_range(0.6) → 중단
    sig = logic.advance(_range(conf=0.5), _close_only(95.0))
    assert sig is None and logic.state_name == "IDLE"


# ---- SL / TP / 사이징 / force_exit ----

def test_initial_stop_loss():
    logic = RangeLogic(PARAMS)
    long_sig = type("S", (), {"side": SignalSide.LONG})()
    short_sig = type("S", (), {"side": SignalSide.SHORT})()
    # k_range_sl=1.5, vol=100 → 거리 150
    assert logic.initial_stop_loss(long_sig, 50000.0, 100.0) == 50000.0 - 150.0
    assert logic.initial_stop_loss(short_sig, 50000.0, 100.0) == 50000.0 + 150.0


def test_take_profit_is_bb_mid():
    logic = RangeLogic(PARAMS)
    _stub_indicators(logic, lower=100.0, mid=110.0, upper=120.0, rsi=50.0)
    # take_profit 은 _bb_mid 사용 (별도 경로) — 실제 캔들로 검증
    closes = list(np.linspace(100, 120, 60))
    tp = logic.take_profit(_candles(closes))
    assert tp is not None
    # 이동평균이므로 마지막 구간 평균 근처
    assert 100.0 < tp < 121.0


def test_update_take_profit_tracks_mid():
    logic = RangeLogic(PARAMS)
    closes = list(np.linspace(100, 120, 60))
    pos = _pos(PositionSide.LONG)
    tp = logic.update_take_profit(_range(), pos, _candles(closes))
    assert tp is not None


def test_position_size_scales_with_confidence():
    logic = RangeLogic(PARAMS)
    sig = type("S", (), {"side": SignalSide.LONG, "confidence": 0.8})()
    size = logic.position_size(sig, 50000.0, 49850.0, 10000.0)
    assert size > 0


def test_force_exit_adverse_trend():
    logic = RangeLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    # range-LONG 인데 하방(short) trend → 적대 → REGIME_EXIT
    dec = logic.force_exit(_trend(RegimeDirection.SHORT), pos)
    assert dec is not None and dec.reason == ExitReason.REGIME_EXIT


def test_no_force_exit_favorable_trend():
    logic = RangeLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    # 우호적(같은방향 long trend) → 유지(None)
    assert logic.force_exit(_trend(RegimeDirection.LONG), pos) is None


def test_no_force_exit_range_persists():
    logic = RangeLogic(PARAMS)
    pos = _pos(PositionSide.LONG)
    assert logic.force_exit(_range(0.9), pos) is None


# ---- I-008: bounded 지표 — 히스토리 길이 무관 (백테=라이브) ----

def test_indicators_bounded_history_invariant():
    """indicator_window 봉만 사용 → 앞쪽 히스토리 길이가 달라도 마지막 값 동일.

    RSI(Wilder RMA)·BB 가 전체 히스토리에 의존하면 백테≠라이브(H4 위반).
    bounded 슬라이스로 마지막 100봉만 보므로 앞에 무엇을 붙여도 결과 불변이어야 한다.
    """
    rng = np.random.default_rng(42)
    tail = list(100 + np.cumsum(rng.normal(0, 1, 150)))  # 공통 꼬리 150봉
    prefix = list(50 + np.cumsum(rng.normal(0, 1, 80)))  # 추가 앞부분

    logic = RangeLogic(PARAMS)  # indicator_window=100
    short = logic._indicators(_candles(tail))
    long = logic._indicators(_candles(prefix + tail))
    assert short is not None and long is not None
    for a, b in zip(short, long):
        assert abs(a - b) < 1e-9


def test_indicators_none_on_warmup():
    logic = RangeLogic(PARAMS)
    # bb_period=20 보다 짧은 캔들 → NaN → None
    assert logic._indicators(_candles([100.0] * 10)) is None
