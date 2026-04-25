"""BacktestEngine end-to-end 테스트.

더미 전략을 등록하고 합성 캔들 데이터로 진입·청산 흐름이 라이브와 동일한
FeeModel/RiskManager를 통해 정상 작동하는지 검증.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.base import StrategyModule
from src.strategy.registry import (
    register_strategy,
    reset_registry_for_testing,
)


# ---- 테스트 fixtures ----


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _make_synthetic_candles(
    n: int = 60,
    start_price: float = 67000.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    """합성 1m 캔들 시리즈 생성."""
    timestamps = pd.date_range(
        "2024-01-01", periods=n, freq="1min", tz="UTC"
    )
    rows = []
    price = start_price
    for _ in range(n):
        o = price
        c = price + drift
        h = max(o, c) + 5
        low = min(o, c) - 5
        rows.append([o, h, low, c, 1.0])
        price = c
    df = pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"], index=timestamps
    )
    df.index.name = "timestamp"
    return df


def _make_config(
    initial_balance: float = 10000.0,
    db_path: str = "data/test.db",
    risk_pct: float = 0.01,
    leverage: int = 5,
) -> dict:
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": db_path},
        "paper": {"initial_balance": initial_balance},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "risk": {
            "max_daily_loss_pct": 0.5,
            "max_drawdown_pct": 0.5,
            "max_position_size_btc": 1.0,
            "max_concurrent_positions": 1,
        },
        "strategies": {"active": ["sl_taker"]},
        "sl_taker": {
            "risk_per_trade_pct": risk_pct,
            "max_leverage": leverage,
        },
    }


# ---- 테스트 전략: 첫 봉에 LONG 진입, SL=-0.5%, TP=+10% (절대 안 닿음) ----


class _SLTakerStrategy(StrategyModule):
    """첫 호출에 LONG 시그널, SL은 진입가 -0.5%, TP는 +10%.
    drift=-100인 합성 캔들에서 SL이 빠르게 hit되어 청산 예상.
    """

    name = "sl_taker"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]

    def __init__(self, params):
        super().__init__(params)
        self._fired = False

    def generate_signal(self, ctx):
        if self._fired:
            return Signal(side=SignalSide.HOLD)
        self._fired = True
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.995  # -0.5%

    def compute_take_profit(self, ctx, signal, sl):
        return ctx.current_price * 1.10  # +10%


# ---- 테스트 전략: TP를 빠르게 hit (drift=+positive에서) ----


class _TPTakerStrategy(StrategyModule):
    name = "tp_taker"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]

    def __init__(self, params):
        super().__init__(params)
        self._fired = False

    def generate_signal(self, ctx):
        if self._fired:
            return Signal(side=SignalSide.HOLD)
        self._fired = True
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.95

    def compute_take_profit(self, ctx, signal, sl):
        return ctx.current_price * 1.005  # +0.5%


# ---- 테스트 케이스 ----


@pytest.mark.asyncio
async def test_sl_hit_closes_with_loss(tmp_path):
    register_strategy(_SLTakerStrategy)
    db_path = str(tmp_path / "bt.db")
    config = _make_config(db_path=db_path, initial_balance=10000)

    eng = BacktestEngine(
        config, start="2024-01-01", end="2024-01-02"
    )
    # 캔들 직접 주입 (HistoricalDataLoader 우회)
    candles = _make_synthetic_candles(n=30, start_price=67000, drift=-50)
    eng.inject_candles({"1m": candles})

    # broker/data_store 초기화 + 잔액 세팅 (initialize의 _load_candles만 우회)
    await eng.broker.initialize()
    balance0 = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(balance0)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert isinstance(result, BacktestResult)
    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade["status"] == "closed"
    assert trade["strategy_name"] == "sl_taker"
    assert trade["exit_reason"] == "sl_hit"
    # SL hit → 손실
    assert trade["pnl"] < 0
    assert result.final_balance < result.initial_balance


@pytest.mark.asyncio
async def test_tp_hit_closes_with_profit(tmp_path):
    register_strategy(_TPTakerStrategy)
    db_path = str(tmp_path / "bt2.db")
    config = _make_config(db_path=db_path, initial_balance=10000)
    config["strategies"]["active"] = ["tp_taker"]
    config["tp_taker"] = config.pop("sl_taker")

    eng = BacktestEngine(
        config, start="2024-01-01", end="2024-01-02"
    )
    candles = _make_synthetic_candles(n=30, start_price=67000, drift=+50)
    eng.inject_candles({"1m": candles})

    await eng.broker.initialize()
    balance0 = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(balance0)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade["exit_reason"] == "tp_hit"
    assert trade["pnl"] > 0
    assert result.win_rate == 100.0


@pytest.mark.asyncio
async def test_engine_shutdown_closes_open_position(tmp_path):
    """청산 신호 없이 백테 종료 시 마지막 캔들로 강제 청산되는지."""
    class _NeverExitStrategy(StrategyModule):
        name = "never_exit"
        entry_timeframe = "1m"
        required_timeframes = ["1m"]

        def __init__(self, params):
            super().__init__(params)
            self._fired = False

        def generate_signal(self, ctx):
            if self._fired:
                return Signal(side=SignalSide.HOLD)
            self._fired = True
            return Signal(side=SignalSide.LONG)

        def compute_stop_loss(self, ctx, s): return ctx.current_price * 0.5
        def compute_take_profit(self, ctx, s, sl): return ctx.current_price * 2.0

    register_strategy(_NeverExitStrategy)
    db_path = str(tmp_path / "bt3.db")
    config = _make_config(db_path=db_path)
    config["strategies"]["active"] = ["never_exit"]
    config["never_exit"] = config.pop("sl_taker")

    eng = BacktestEngine(config, start="2024-01-01", end="2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=10, drift=+10)})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert result.num_trades == 1
    assert result.trades[0]["exit_reason"] == "engine_shutdown"


@pytest.mark.asyncio
async def test_no_active_strategies_returns_initial_balance(tmp_path):
    """I-010: 활성 전략 0개일 때 final_balance = initial_balance fallback."""
    config = _make_config(db_path=str(tmp_path / "bt.db"), initial_balance=10000)
    config["strategies"] = {"active": []}
    config.pop("sl_taker", None)

    eng = BacktestEngine(config, start="2024-01-01", end="2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=10)})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # 활성 전략 0이면 timeframes 비어 master 없음 → run 즉시 종료
    assert result.num_trades == 0
    assert result.final_balance == 10000.0  # initial로 fallback
    assert result.total_pnl == 0.0
    assert result.total_pnl_pct == 0.0


@pytest.mark.asyncio
async def test_write_reports_creates_all_files(tmp_path, monkeypatch):
    """I-011: write_reports가 5종 파일을 디스크에 생성."""
    register_strategy(_TPTakerStrategy)
    config = _make_config(db_path=str(tmp_path / "bt.db"), initial_balance=10000)
    config["strategies"]["active"] = ["tp_taker"]
    config["tp_taker"] = config.pop("sl_taker")

    # REPORT_BASE를 tmp 경로로 우회
    from src.backtest import engine as bt_module
    monkeypatch.setattr(
        bt_module, "REPORT_BASE", tmp_path / "00_Working"
    )

    eng = BacktestEngine(config, start="2024-01-01", end="2024-01-02")
    eng.inject_candles(
        {"1m": _make_synthetic_candles(n=30, drift=+50)}
    )
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    out_dir = eng.write_reports(config_path="config/test_config.yaml")
    await eng.shutdown()

    assert out_dir.exists()
    assert out_dir.name == "test_config"
    parent = out_dir.parent
    assert parent.name.startswith("backtest_") or "backtest" in parent.name
    # 5종 파일
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "equity_curve.png").exists()

    # config_snapshot 자격증명 제거 확인
    import yaml as _yaml
    with open(out_dir / "config_snapshot.yaml", encoding="utf-8") as f:
        snap = _yaml.safe_load(f)
    assert "api_key" not in snap.get("exchange", {})

    # metrics.json 구조
    import json as _json
    with open(out_dir / "metrics.json", encoding="utf-8") as f:
        metrics = _json.load(f)
    assert "integrated" in metrics
    assert "by_strategy_name" in metrics
    assert "tp_taker" in metrics["by_strategy_name"]
    assert "by_exit_reason" in metrics
    assert "tp_hit" in metrics["by_exit_reason"]
    assert "by_direction" in metrics


@pytest.mark.asyncio
async def test_write_reports_empty_trades(tmp_path, monkeypatch):
    """무거래 백테에서도 빈 trades.csv / equity_curve.csv 헤더만 출력."""
    config = _make_config(db_path=str(tmp_path / "bt.db"))
    config["strategies"] = {"active": []}
    config.pop("sl_taker", None)

    from src.backtest import engine as bt_module
    monkeypatch.setattr(
        bt_module, "REPORT_BASE", tmp_path / "00_Working"
    )

    eng = BacktestEngine(config, start="2024-01-01", end="2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=5)})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    out_dir = eng.write_reports(config_path="config/empty.yaml")
    await eng.shutdown()

    # 5종 파일 모두 존재 (trades/equity는 헤더만)
    for fname in (
        "trades.csv", "equity_curve.csv", "metrics.json", "config_snapshot.yaml"
    ):
        assert (out_dir / fname).exists(), fname
    # equity_curve.png는 빈 데이터일 때 안 만들어질 수 있음 — 강제 검증 X
    trades_content = (out_dir / "trades.csv").read_text(encoding="utf-8")
    assert "id,strategy_name" in trades_content


# ---- 다중 전략 (C) 배타 경합 정책 검증 ----


class _FirstLongStrategy(StrategyModule):
    """첫 호출에만 LONG, 이후 HOLD. 한 번 발사하면 끝."""

    name = "first_long"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]

    def __init__(self, params):
        super().__init__(params)
        self._fired = False
        self.signal_calls = 0

    def generate_signal(self, ctx):
        self.signal_calls += 1
        if self._fired:
            return Signal(side=SignalSide.HOLD)
        self._fired = True
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, s): return ctx.current_price * 0.995
    def compute_take_profit(self, ctx, s, sl): return ctx.current_price * 1.10


class _AlwaysLongStrategy(StrategyModule):
    """슬롯이 비기만 하면 매번 LONG."""

    name = "always_long"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]

    def __init__(self, params):
        super().__init__(params)
        self.signal_calls = 0

    def generate_signal(self, ctx):
        self.signal_calls += 1
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, s): return ctx.current_price * 0.995
    def compute_take_profit(self, ctx, s, sl): return ctx.current_price * 1.10


def _multi_strategy_config(db_path: str, *, active: list[str]) -> dict:
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": db_path},
        "paper": {"initial_balance": 10000},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "risk": {
            "max_daily_loss_pct": 0.5,
            "max_drawdown_pct": 0.5,
            "max_position_size_btc": 1.0,
            "max_concurrent_positions": 1,
        },
        "strategies": {"active": active},
        "first_long": {"risk_per_trade_pct": 0.01, "max_leverage": 5},
        "always_long": {"risk_per_trade_pct": 0.01, "max_leverage": 5},
    }


@pytest.mark.asyncio
async def test_multi_strategy_priority_first_wins(tmp_path):
    """우선순위 첫 전략이 슬롯 선점, 두 번째 전략은 슬롯 비기 전까지 미평가."""
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)

    config = _multi_strategy_config(
        str(tmp_path / "bt.db"), active=["first_long", "always_long"]
    )
    candles = _make_synthetic_candles(n=40, start_price=67000, drift=-50)

    eng = BacktestEngine(
        config,
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    assert result.num_trades >= 2, "필요 시 drift 조정으로 SL 여러 번 발생해야 함"

    by_strategy: dict[str, list] = {}
    for t in result.trades:
        by_strategy.setdefault(t["strategy_name"], []).append(t)

    # first_long: 첫 호출 한 번만 발사 (이후 HOLD)
    assert "first_long" in by_strategy
    assert len(by_strategy["first_long"]) == 1

    # always_long: first_long 청산 후 슬롯 비면 그때부터 진입
    assert "always_long" in by_strategy
    assert len(by_strategy["always_long"]) >= 1

    # 시간 순서: first_long이 가장 먼저 진입
    first_entry = by_strategy["first_long"][0]["entry_time"]
    earliest_always = min(t["entry_time"] for t in by_strategy["always_long"])
    assert first_entry < earliest_always


@pytest.mark.asyncio
async def test_multi_strategy_lower_priority_skipped_when_slot_full(tmp_path):
    """(C) Ignore 정책: 슬롯 차있을 때 낮은 우선순위 전략의 generate_signal 호출 안 됨."""
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)

    config = _multi_strategy_config(
        str(tmp_path / "bt.db"), active=["first_long", "always_long"]
    )
    candles = _make_synthetic_candles(n=40, start_price=67000, drift=-50)

    eng = BacktestEngine(
        config,
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    await eng.shutdown()

    first_inst = next(s for s in eng.strategies if s.name == "first_long")
    always_inst = next(s for s in eng.strategies if s.name == "always_long")

    # first_long은 모든 봉에서 호출 (슬롯 비어있을 때 첫 순서이므로)
    assert first_inst.signal_calls > 0
    # always_long은 first_long이 슬롯 점유 중일 때 호출 안 됨 →
    # first_long 호출 횟수보다 적어야 함
    assert always_inst.signal_calls < first_inst.signal_calls
    # 슬롯이 한 번이라도 비어 always_long 평가 받았어야 의미 있는 검증
    assert always_inst.signal_calls > 0


@pytest.mark.asyncio
async def test_multi_strategy_swapped_priority(tmp_path):
    """우선순위 순서를 뒤집으면 always_long이 슬롯 독점, first_long은 전혀 진입 못 함."""
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)

    config = _multi_strategy_config(
        str(tmp_path / "bt.db"),
        active=["always_long", "first_long"],  # 순서 뒤집음
    )
    candles = _make_synthetic_candles(n=40, start_price=67000, drift=-50)

    eng = BacktestEngine(
        config,
        start=candles.index[0].to_pydatetime(),
        end=candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    by_strategy: dict[str, list] = {}
    for t in result.trades:
        by_strategy.setdefault(t["strategy_name"], []).append(t)

    # always_long이 우선이므로 모든 거래가 always_long
    assert "always_long" in by_strategy
    assert "first_long" not in by_strategy
    assert len(by_strategy["always_long"]) >= 2


def test_backtest_result_metrics():
    r = BacktestResult(
        initial_balance=10000,
        final_balance=10500,
        equity_curve=[
            (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000),
            (datetime(2024, 1, 2, tzinfo=timezone.utc), 11000),
            (datetime(2024, 1, 3, tzinfo=timezone.utc), 9000),  # peak 11000 → -18%
            (datetime(2024, 1, 4, tzinfo=timezone.utc), 10500),
        ],
        trades=[
            {"pnl": 1500},
            {"pnl": -1000},
            {"pnl": 0},  # 무승부 — winner도 loser도 아님
        ],
    )
    assert r.total_pnl == 500
    assert r.total_pnl_pct == 5.0
    assert r.num_trades == 3
    assert r.num_winners == 1
    assert r.num_losers == 1
    assert r.win_rate == pytest.approx(33.33, abs=0.01)
    # 11000 → 9000 = -18.18%
    assert r.max_drawdown_pct == pytest.approx(18.18, abs=0.01)
