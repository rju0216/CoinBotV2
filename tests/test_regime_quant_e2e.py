"""RegimeQuantStrategy E2E 백테 통합 테스트 (S3).

실 BacktestEngine 루프 + tiny Gaussian artifact + 합성 1h 캔들로 검증:
  - 정합성(규칙10): trades pnl 합 ↔ metrics total_pnl ↔ equity 변화 일치
  - 레짐 스왑(I-002④): 적대 전환 → force_exit(REGIME_EXIT) + 같은 봉 빈슬롯 재진입
  - churn(I-006): 트레일링 손절 직후 재진입 (관대진입 DD-1 churn 계측)

결정론: plugin._contract 를 직전 마감 봉 ts 기준으로 stub (HMM 학습 결과 비결정성 회피).
진입가=다음 봉 시초가, SL/TP 캔들 체결 등 실엔진 흐름은 그대로 탄다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.plugins.regime_quant import RegimeQuantStrategy
from src.strategy.regime.artifact import save_model
from src.strategy.regime.contract import (
    Contract as RegimeContract,
    RegimeDirection,
    RegimeType,
)
from src.strategy.regime.training import build_model
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.test_regime_quant_plugin import PARAMS, _synth_candles  # 순수 상수·함수


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("regime_models_e2e")
    model = build_model(
        _synth_candles(300),
        k=2,
        tau=0.5,
        valid_period=("2020-01-01", "2035-01-01"),
        n_init=1,
        n_iter=8,
        seed=0,
    )
    save_model(model, d / "m0.json")
    return str(d)


def _config(model_dir, db_path):
    return {
        "exchange": {"symbol": "BTC/USDT:USDT", "leverage": 5},
        "database": {"path": db_path},
        "paper": {"initial_balance": 10000.0},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0,
                       "funding_enabled": False},
        "strategies": {"active": ["regime_quant"]},
        "regime_quant": {**PARAMS, "model_dir": model_dir},
    }


def _candles_1h(closes, hl_pad=1.0):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, dtype=float, index=idx)
    o = c.shift(1)
    o.iloc[0] = c.iloc[0]
    oc = pd.concat([o, c], axis=1)
    return pd.DataFrame(
        {
            "open": o,
            "high": oc.max(axis=1) + hl_pad,
            "low": oc.min(axis=1) - hl_pad,
            "close": c,
            "volume": 1.0,
        },
        index=idx,
    )


def _ct(type_, direction, conf=0.8, vol=50.0):
    return RegimeContract(type=type_, direction=direction, confidence=conf, volatility=vol)


async def _setup(eng):
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.account_tracker.set_initial_balance(bal)


def _assert_consistency(eng, result):
    """규칙10: trades pnl 합 ↔ metrics total_pnl ↔ equity 변화 3자 일치."""
    pnl_sum = sum(t["pnl"] for t in result.trades)
    metrics_total = eng._build_metrics()["integrated"]["total_pnl"]
    eq_delta = (
        eng.equity_curve[-1][1] - eng.account_tracker.initial_balance
        if eng.equity_curve else 0.0
    )
    assert abs(pnl_sum - result.total_pnl) < 1e-6
    assert abs(round(pnl_sum, 2) - metrics_total) < 1e-2
    assert abs(eq_delta - result.total_pnl) < 1e-6


# ---- E2E-1: 정합성 (단일 거래) ----

@pytest.mark.asyncio
async def test_e2e_consistency_single_trade(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    candles = _candles_1h([67000.0 + (i % 3) * 5 for i in range(30)], hl_pad=2.0)
    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    # 항상 trend long (SL 멀어 안 닿음 → engine_shutdown 까지 단일 포지션)
    eng.strategies[0]._contract = lambda ctx: _ct(
        RegimeType.TREND, RegimeDirection.LONG, vol=50.0
    )
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert result.num_trades == 1
    assert result.trades[0]["side"] == "long"
    assert result.trades[0]["exit_reason"] == "engine_shutdown"
    assert result.trades[0]["status"] == "closed"
    _assert_consistency(eng, result)


# ---- E2E-2: 레짐 스왑 (I-002④) ----

@pytest.mark.asyncio
async def test_e2e_regime_swap_same_bar(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    closes = [67000.0 + (i % 3) * 5 for i in range(20)]
    candles = _candles_1h(closes, hl_pad=2.0)
    swap_idx = 8  # 직전 봉 인덱스가 이 값 이상이면 적대(short) 전환

    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    def stub(ctx):
        df = ctx.candles.get("1h")
        if df is None or len(df) == 0:
            return None
        i = candles.index.get_loc(df.index[-1])  # 직전 마감 봉 인덱스
        d = RegimeDirection.LONG if i < swap_idx else RegimeDirection.SHORT
        return _ct(RegimeType.TREND, d, vol=50.0)

    eng.strategies[0]._contract = stub
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # 적대 전환 청산(REGIME_EXIT) + short 재진입이 같은 봉에 발생
    regime_exits = [t for t in result.trades if t["exit_reason"] == "regime_exit"]
    assert len(regime_exits) >= 1
    long_exit = regime_exits[0]
    assert long_exit["side"] == "long"

    shorts = [t for t in result.trades if t["side"] == "short"]
    assert len(shorts) >= 1
    # 같은 봉 스왑: long 청산 시각 == short 진입 시각
    assert long_exit["exit_time"] == shorts[0]["entry_time"]
    assert long_exit["logic"] == "trend"  # logic 컬럼 기록 검증(trend 디스패치)
    _assert_consistency(eng, result)


# ---- E2E-3: churn (I-006) ----

@pytest.mark.asyncio
async def test_e2e_churn_repeated_sl_reentry(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    # 계단 하락 (봉간 -300) → 진입 직후 다음 봉 SL(거리 200) 히트 반복
    candles = _candles_1h([67000.0 - i * 300 for i in range(15)], hl_pad=1.0)
    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    eng.strategies[0]._contract = lambda ctx: _ct(
        RegimeType.TREND, RegimeDirection.LONG, vol=100.0  # SL 거리 = 2*100 = 200
    )
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # churn: 진입→SL→재진입 반복 → 다수 거래
    assert result.num_trades >= 3
    sl_hits = [t for t in result.trades if t["exit_reason"] == "sl_hit"]
    assert len(sl_hits) >= 2  # 트레일링 손절(고정 SL) 반복
    # 관대진입(DD-1): 같은 레짐 유지 시 손절 직후 재진입 (가드 없음 = churn)
    assert all(t["side"] == "long" for t in result.trades)
    _assert_consistency(eng, result)


# ---- E2E-4: range 적대청산 (range 로직 실엔진 디스패치) ----

@pytest.mark.asyncio
async def test_e2e_range_adverse_exit(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    # 하단(66000) 진입 → 중심선(≈66950)·SL 모두 안 닿게 유지 → 보유 중 적대 전환 받기.
    # (횡보면 진입가≈중심선이라 진입 직후 TP 가 먼저 체결되어 적대 청산을 못 봄.)
    candles = _candles_1h([67000.0] * 24 + [66000.0] * 6, hl_pad=1.0)
    trigger_idx = 24  # 직전봉(=66000)에서 LONG 트리거 (BB warmup 충족)
    swap_idx = 27  # 직전봉이 이 값 이상이면 적대(trend short) 전환

    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)
    plugin = eng.strategies[0]

    def stub_contract(ctx):
        df = ctx.candles.get("1h")
        if df is None or len(df) == 0:
            return None
        i = candles.index.get_loc(df.index[-1])
        if i >= swap_idx:
            return _ct(RegimeType.TREND, RegimeDirection.SHORT, vol=50.0)  # 적대
        return _ct(RegimeType.RANGE, RegimeDirection.NONE, vol=50.0)

    def stub_advance(contract, df):
        i = candles.index.get_loc(df.index[-1])
        if i == trigger_idx:  # range 진입 트리거 강제
            return Signal(side=SignalSide.LONG, confidence=0.8,
                          meta={"volatility": 50.0})
        return None

    plugin._contract = stub_contract
    plugin.range.advance = stub_advance
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # range long 진입(advance 경유) → 적대 trend short → range.force_exit → REGIME_EXIT
    regime_exits = [t for t in result.trades if t["exit_reason"] == "regime_exit"]
    assert len(regime_exits) >= 1
    assert regime_exits[0]["side"] == "long"
    # 'range 로직' 입증: logic 컬럼 == range (+ range 만 TP(BB 중심선) 등록).
    assert regime_exits[0]["logic"] == "range"
    assert regime_exits[0]["take_profit"] is not None
    _assert_consistency(eng, result)


# ---- E2E-5: service 통합 스모크 (stub 없이 실 HMM filtering) ----

@pytest.mark.asyncio
async def test_e2e_service_integration_smoke(model_dir, tmp_path):
    """stub 없이 실 RegimeService.get_contract(HMM filtering)→매매층 전체 경로 1회 완주.

    진입은 HMM 결과라 비결정 → num_trades 단언 없이 '예외 없이 완주 + 정합성 +
    service 가 warmup 후 실제 Contract 를 산출(매매층에 전달)' 만 검증.
    """
    register_strategy(RegimeQuantStrategy)
    candles = _synth_candles(250)  # >= need(window100+warmup48=148)
    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    # service.get_contract 호출 추적 (stub 없음 — 실 filtering 경로)
    plugin = eng.strategies[0]
    orig = plugin.service.get_contract
    seen = {"non_none": 0}

    def traced(df):
        r = orig(df)
        if r is not None:
            seen["non_none"] += 1
        return r

    plugin.service.get_contract = traced
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert seen["non_none"] > 0  # warmup 지나 실제 Contract 산출 → service→매매 배선 실행
    _assert_consistency(eng, result)


# ---- E2E-6: trend 트레일 익절 (트레일링이 SL 을 끌어올려 익절 청산) ----

@pytest.mark.asyncio
async def test_e2e_trend_trailing_profit(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    # 진입 후 상승 → 트레일 SL 전진(진입가 위로) → 마지막 급락 봉이 그 SL 관통 → 익절
    candles = _candles_1h(
        [67000, 67000, 68000, 69000, 70000, 71000, 72000, 65000], hl_pad=1.0
    )
    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    eng.strategies[0]._contract = lambda ctx: _ct(
        RegimeType.TREND, RegimeDirection.LONG, vol=100.0  # SL 200, 트레일 250
    )
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # 첫 거래: 트레일 SL 이 진입가(67000)보다 높이 올라가 sl_hit 으로 익절
    first = result.trades[0]
    assert first["exit_reason"] == "sl_hit"
    assert first["pnl"] > 0  # 진입가보다 높은 SL 에서 청산 = 트레일 익절
    # exit_price = 끌어올린 SL (진입가보다 높음). trade["stop_loss"] 는 진입 시 초기값이라
    # 트레일 결과가 아님 → 청산가로 트레일 익절을 입증.
    assert first["exit_price"] > first["entry_price"]
    _assert_consistency(eng, result)


# ---- E2E-7: range 이동 중심선 TP 체결 ----

@pytest.mark.asyncio
async def test_e2e_range_tp_hit(model_dir, tmp_path):
    register_strategy(RegimeQuantStrategy)
    # 앞 26봉 평탄(67000) + 급락(66000) 봉에서 range long 진입 → BB 중심선(≈66950) 위로
    # 복귀 → 이동 중심선 TP 체결.
    closes = [67000.0] * 26 + [66000.0] + [67000.0] * 4
    candles = _candles_1h(closes, hl_pad=1.0)
    trigger_idx = 26  # 직전봉(=66000 급락봉)에서 LONG 트리거 강제

    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)
    plugin = eng.strategies[0]
    plugin._contract = lambda ctx: _ct(
        RegimeType.RANGE, RegimeDirection.NONE, vol=50.0
    )

    def stub_advance(contract, df):
        i = candles.index.get_loc(df.index[-1])
        if i == trigger_idx:
            return Signal(side=SignalSide.LONG, confidence=0.8,
                          meta={"volatility": 50.0})
        return None

    plugin.range.advance = stub_advance
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # range long 진입(@66000) → 가격 복귀(67000)로 BB 중심선 도달 → TP_HIT 익절
    tp_hits = [t for t in result.trades if t["exit_reason"] == "tp_hit"]
    assert len(tp_hits) >= 1
    assert tp_hits[0]["side"] == "long"
    assert tp_hits[0]["pnl"] > 0
    _assert_consistency(eng, result)


# ---- E2E-8: NONE 매핑 = 무매매 (추세-단독 gen1, O-7 (가)) ----
# "플러그인 무변경" 주장(NONE 은 is_trend/is_range 어디에도 안 걸림)을 실엔진으로 실증.

@pytest.mark.asyncio
async def test_e2e_none_contract_no_entry(model_dir, tmp_path):
    """NONE contract 지속 → 진입 0 (NONE = 관망/무매매)."""
    register_strategy(RegimeQuantStrategy)
    candles = _candles_1h([67000.0 + (i % 3) * 5 for i in range(30)], hl_pad=2.0)
    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)
    eng.strategies[0]._contract = lambda ctx: _ct(
        RegimeType.NONE, RegimeDirection.NONE, vol=50.0
    )
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert result.num_trades == 0  # NONE = 무매매 (진입도 range 도 없음)
    _assert_consistency(eng, result)


@pytest.mark.asyncio
async def test_e2e_none_freezes_held_trend(model_dir, tmp_path):
    """보유 trend 중 NONE 전환 → 트레일 동결·force_exit 없음 → shutdown 까지 보유(스펙 §3.3)."""
    register_strategy(RegimeQuantStrategy)
    candles = _candles_1h([67000.0 + (i % 3) * 5 for i in range(20)], hl_pad=2.0)
    none_idx = 8  # 직전봉 인덱스 >= 이면 NONE 전환

    eng = BacktestEngine(
        _config(model_dir, str(tmp_path / "bt.db")),
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1h": candles})
    await _setup(eng)

    def stub(ctx):
        df = ctx.candles.get("1h")
        if df is None or len(df) == 0:
            return None
        i = candles.index.get_loc(df.index[-1])
        if i < none_idx:
            return _ct(RegimeType.TREND, RegimeDirection.LONG, vol=50.0)
        return _ct(RegimeType.NONE, RegimeDirection.NONE, vol=50.0)

    eng.strategies[0]._contract = stub
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # trend long 진입 후 NONE 전환 → 청산(force_exit) 안 일어나고 shutdown 까지 단일 보유
    assert result.num_trades == 1
    assert result.trades[0]["side"] == "long"
    assert result.trades[0]["exit_reason"] == "engine_shutdown"
    _assert_consistency(eng, result)
