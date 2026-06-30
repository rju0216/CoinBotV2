"""RegimeQuantStrategy 디스패처 플러그인 테스트 (S2).

- artifact 통합 로딩 + 무예외 동작(smoke)
- contract→logic 디스패치 정확성 (trend/range 분기, position.meta stash)
- artifact 부재 에러
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import Position, Signal, StrategyContext
from src.strategy.plugins.regime_quant import RegimeQuantStrategy
from src.strategy.regime.artifact import save_model
from src.strategy.regime.contract import (
    Contract,
    RegimeDirection,
    RegimeType,
)
from src.strategy.regime.training import build_model

PARAMS = {
    "theta_trend": 0.6,
    "k_trend_sl": 2.0,
    "k_trend_trail": 2.5,
    "theta_range": 0.6,
    "k_range_sl": 1.5,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_period": 20,
    "bb_std": 2.0,
    "expiry_atr_mult": 1.0,
    "risk_per_trade_pct": 0.01,
    "max_leverage": 5,
    "filter_window": 100,
    "indicator_window": 100,
    "atr_period": 24,
}


def _synth_candles(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """tiny Gaussian artifact 1개 사전학습 → tmp 디렉토리 (모듈 1회)."""
    d = tmp_path_factory.mktemp("regime_models")
    candles = _synth_candles(300)
    model = build_model(
        candles,
        k=2,
        tau=0.5,
        valid_period=("2020-01-01", "2035-01-01"),
        n_init=1,
        n_iter=8,
        seed=0,
    )
    save_model(model, d / "m0.json")
    return str(d)


def _ctx(candles, price=100.0, balance=10000.0):
    return StrategyContext(
        candles={"1h": candles},
        current_price=price,
        balance=balance,
        position=None,
        is_slot_occupied=False,
        params={},
        now=datetime.now(timezone.utc),
    )


def _pos(side, entry=100.0, sl=None, logic="trend"):
    p = Position(
        side=side,
        size=1.0,
        entry_price=entry,
        entry_time=datetime.now(timezone.utc),
        strategy_name="regime_quant",
        stop_loss=sl,
    )
    p.meta["logic"] = logic
    return p


def _contract(type_, direction, conf, vol=2.0):
    return Contract(type=type_, direction=direction, confidence=conf, volatility=vol)


# ---- artifact 로딩 ----

def test_missing_model_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RegimeQuantStrategy({**PARAMS, "model_dir": str(tmp_path / "nope")})


def test_plugin_loads_and_runs(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    sig = plugin.generate_signal(_ctx(_synth_candles(300)))
    assert isinstance(sig, Signal)  # 예외 없이 Signal 반환 (HOLD 또는 actionable)


def test_class_attributes(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    assert plugin.name == "regime_quant"
    assert plugin.entry_timeframe == "1h"
    assert plugin.sl_tp_fill_priority == "sl_first"
    assert plugin.allow_entry(_ctx(_synth_candles(300))) is True


# ---- trend 디스패치 ----

def test_trend_dispatch_full(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    candles = _synth_candles(300)
    ctx = _ctx(candles, price=100.0)

    # contract 강제 (trend/long, conf≥θ, ATR=2.0)
    plugin._contract = lambda c: _contract(
        RegimeType.TREND, RegimeDirection.LONG, 0.8, vol=2.0
    )
    sig = plugin.generate_signal(ctx)
    assert sig.side == SignalSide.LONG
    assert sig.meta["logic"] == "trend"

    sl = plugin.compute_stop_loss(ctx, sig)
    assert sl == 100.0 - 2.0 * 2.0  # k_trend_sl·ATR
    assert plugin.compute_take_profit(ctx, sig, sl) is None  # 추세 TP 없음
    assert plugin.compute_position_size(ctx, sig, sl) > 0

    # position.meta stash
    pos = _pos(PositionSide.LONG, entry=100.0, sl=sl, logic="x")
    plugin.on_position_opened(pos)
    assert pos.meta["logic"] == "trend"
    assert pos.meta["entry_contract"]["type"] == "trend"

    # 보유 중: 동방향 trend → 트레일링 갱신(가격 상승), TP None
    pos.meta["logic"] = "trend"
    ctx_up = _ctx(candles, price=110.0)  # ATR2, trail2.5 → 110-5=105 > sl(96) → 갱신
    assert plugin.update_stop_loss(ctx_up, pos) == 105.0
    assert plugin.update_take_profit(ctx_up, pos) is None

    # 적대 전환 → REGIME_EXIT
    plugin._contract = lambda c: _contract(
        RegimeType.TREND, RegimeDirection.SHORT, 0.8, vol=2.0
    )
    dec = plugin.should_force_exit(ctx, pos)
    assert dec is not None and dec.reason == ExitReason.REGIME_EXIT


def test_trend_blocked_below_theta(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._contract = lambda c: _contract(
        RegimeType.TREND, RegimeDirection.LONG, 0.5, vol=2.0
    )
    sig = plugin.generate_signal(_ctx(_synth_candles(300)))
    assert sig.side == SignalSide.HOLD


# ---- range 디스패치 ----

def test_range_dispatch_full(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    candles = _synth_candles(300)
    ctx = _ctx(candles, price=100.0)

    plugin._contract = lambda c: _contract(
        RegimeType.RANGE, RegimeDirection.NONE, 0.8, vol=2.0
    )
    # 상태기계 트리거를 강제 (advance 가 LONG Signal 반환)
    plugin.range.advance = lambda contract, df: Signal(
        side=SignalSide.LONG, confidence=0.8, meta={"volatility": 2.0}
    )
    sig = plugin.generate_signal(ctx)
    assert sig.side == SignalSide.LONG
    assert sig.meta["logic"] == "range"

    sl = plugin.compute_stop_loss(ctx, sig)
    assert sl == 100.0 - 1.5 * 2.0  # k_range_sl·ATR
    tp = plugin.compute_take_profit(ctx, sig, sl)
    assert tp is not None  # range TP = BB 중심선 (float)

    pos = _pos(PositionSide.LONG, entry=100.0, sl=sl, logic="range")
    # range 는 트레일링 없음(SL 고정)
    assert plugin.update_stop_loss(ctx, pos) is None
    # range 이동 중심선 TP 추적 (float)
    assert plugin.update_take_profit(ctx, pos) is not None

    # 적대(하방 trend) → REGIME_EXIT
    plugin._contract = lambda c: _contract(
        RegimeType.TREND, RegimeDirection.SHORT, 0.8, vol=2.0
    )
    dec = plugin.should_force_exit(ctx, pos)
    assert dec is not None and dec.reason == ExitReason.REGIME_EXIT


# ---- contract None (warmup/모델없음) → 안전 ----

def test_contract_none_holds(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._contract = lambda c: None
    ctx = _ctx(_synth_candles(300))
    assert plugin.generate_signal(ctx).side == SignalSide.HOLD
    pos = _pos(PositionSide.LONG, logic="trend")
    assert plugin.update_stop_loss(ctx, pos) is None
    assert plugin.update_take_profit(ctx, pos) is None
    assert plugin.should_force_exit(ctx, pos) is None


# ---- _pending_meta 생명주기 / reset / 방어 ----

def test_pending_meta_consumed_and_reset(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._contract = lambda c: _contract(
        RegimeType.TREND, RegimeDirection.LONG, 0.8, vol=2.0
    )
    plugin.generate_signal(_ctx(_synth_candles(300)))
    assert plugin._pending_meta is not None
    pos = _pos(PositionSide.LONG, logic="x")
    plugin.on_position_opened(pos)
    assert pos.meta["logic"] == "trend"
    assert plugin._pending_meta is None  # 소비 후 리셋


def test_pending_meta_stale_overwritten(model_dir):
    """진입 실패로 stale stash 가 남아도, 새 진입의 generate_signal 이 덮어쓴다."""
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._pending_meta = {  # 이전 trend 진입 실패 잔재
        "logic": "trend",
        "entry_contract": {"type": "trend", "direction": "long",
                           "confidence": 0.8, "volatility": 2.0},
    }
    plugin._contract = lambda c: _contract(
        RegimeType.RANGE, RegimeDirection.NONE, 0.8, vol=2.0
    )
    plugin.range.advance = lambda contract, df: Signal(
        side=SignalSide.LONG, confidence=0.8, meta={"volatility": 2.0}
    )
    plugin.generate_signal(_ctx(_synth_candles(300)))
    assert plugin._pending_meta["logic"] == "range"  # 덮어써짐
    pos = _pos(PositionSide.LONG, logic="x")
    plugin.on_position_opened(pos)
    assert pos.meta["logic"] == "range"


def test_on_position_opened_defensive_default(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._pending_meta = None  # stash 없이 열린 비정상 경로 (라이브 복원 등)
    pos = Position(
        side=PositionSide.LONG, size=1.0, entry_price=100.0,
        entry_time=datetime.now(timezone.utc), strategy_name="regime_quant",
    )
    plugin.on_position_opened(pos)
    assert pos.meta["logic"] == "trend"  # setdefault 방어값


def test_on_position_closed_resets_range(model_dir):
    from src.strategy.regime.range_logic import _RangeState

    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin.range._state = _RangeState.ARMED_LONG
    plugin.on_position_closed(_pos(PositionSide.LONG, logic="range"), 10.0)
    assert plugin.range.state_name == "IDLE"


def test_position_size_none_stop_returns_zero(model_dir):
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    sig = Signal(side=SignalSide.LONG, confidence=0.8,
                 meta={"logic": "trend", "volatility": 2.0})
    assert plugin.compute_position_size(_ctx(_synth_candles(300)), sig, None) == 0.0


def test_hold_signals_are_distinct(model_dir):
    """_HOLD 싱글톤 제거 회귀 가드 — HOLD 마다 별 객체라 meta 오염 격리."""
    plugin = RegimeQuantStrategy({**PARAMS, "model_dir": model_dir})
    plugin._contract = lambda c: None
    ctx = _ctx(_synth_candles(300))
    h1 = plugin.generate_signal(ctx)
    h2 = plugin.generate_signal(ctx)
    assert h1 is not h2
    h1.meta["note"] = "x"
    assert "note" not in h2.meta
