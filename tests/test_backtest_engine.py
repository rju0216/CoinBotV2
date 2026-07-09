"""BacktestEngine end-to-end 테스트.

더미 전략 + 합성 캔들로 진입·청산 흐름과 FeeModel/AccountTracker 정산을 검증.
엔진은 동기이며 체결은 캔들 가격으로 시뮬한다 (외부 브로커 없음).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _make_synthetic_candles(
    n: int = 60, start_price: float = 67000.0, drift: float = 0.0
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
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


def _make_config(active: list[str], initial_balance: float = 10000.0) -> dict:
    cfg = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "backtest": {"initial_balance": initial_balance},
        "strategies": {"active": active},
    }
    for name in active:
        cfg[name] = {}
    return cfg


def _run(eng: BacktestEngine) -> None:
    """네트워크 로딩 없이(inject_candles 후) run 을 구동."""
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()


# ---- 테스트 전략 ----


class _SLTakerStrategy(StubStrategy):
    """첫 봉 LONG, SL=-0.5%, TP=+10%. drift<0 캔들에서 SL 빠르게 hit."""

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
        return ctx.current_price * 0.995

    def compute_take_profit(self, ctx, signal, sl):
        return ctx.current_price * 1.10


class _TPTakerStrategy(_SLTakerStrategy):
    name = "tp_taker"

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.95

    def compute_take_profit(self, ctx, signal, sl):
        return ctx.current_price * 1.005  # +0.5% — drift>0 에서 빠르게 hit


# ---- 테스트 케이스 ----


def test_sl_hit_closes_with_loss():
    register_strategy(_SLTakerStrategy)
    eng = BacktestEngine(_make_config(["sl_taker"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=30, drift=-50)})
    _run(eng)

    assert len(eng.trades) == 1
    trade = eng.trades[0]
    assert trade["status"] == "closed"
    assert trade["strategy_name"] == "sl_taker"
    assert trade["exit_reason"] == "sl_hit"
    assert trade["pnl"] < 0
    assert eng.balance < 10000.0


def test_tp_hit_closes_with_profit():
    register_strategy(_TPTakerStrategy)
    eng = BacktestEngine(_make_config(["tp_taker"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=30, drift=+50)})
    _run(eng)

    assert len(eng.trades) == 1
    assert eng.trades[0]["exit_reason"] == "tp_hit"
    assert eng.trades[0]["pnl"] > 0
    assert eng.summary()["win_rate_pct"] == 100.0


def test_engine_shutdown_closes_open_position():
    """청산 신호 없이 백테 종료 시 마지막 캔들로 강제 청산."""

    class _NeverExitStrategy(_SLTakerStrategy):
        name = "never_exit"

        def compute_stop_loss(self, ctx, signal):
            return ctx.current_price * 0.5

        def compute_take_profit(self, ctx, signal, sl):
            return ctx.current_price * 2.0

    register_strategy(_NeverExitStrategy)
    eng = BacktestEngine(_make_config(["never_exit"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=10, drift=+10)})
    _run(eng)

    assert len(eng.trades) == 1
    assert eng.trades[0]["exit_reason"] == "engine_shutdown"


def test_no_active_strategies_no_trades():
    eng = BacktestEngine(_make_config([]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=10)})
    _run(eng)

    s = eng.summary()
    assert s["num_trades"] == 0
    assert s["final_balance"] == 10000.0
    assert s["total_pnl"] == 0.0


def test_summary_metrics():
    """summary() 지표 산출 (거래·drawdown)."""
    register_strategy(_TPTakerStrategy)
    eng = BacktestEngine(_make_config(["tp_taker"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=40, drift=+50)})
    _run(eng)

    s = eng.summary()
    assert s["num_trades"] >= 1
    assert s["num_winners"] >= 1
    assert s["final_balance"] > s["initial_balance"]
    assert s["total_pnl"] == pytest.approx(
        s["final_balance"] - s["initial_balance"], abs=0.01
    )


def test_write_reports_creates_three_files(tmp_path):
    register_strategy(_TPTakerStrategy)
    eng = BacktestEngine(_make_config(["tp_taker"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=30, drift=+50)})
    _run(eng)

    out_dir = eng.write_reports(out_dir=tmp_path / "reports")
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "metrics.json").exists()

    with open(out_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "total_pnl" in metrics
    assert metrics["num_trades"] >= 1


def test_write_reports_empty_trades(tmp_path):
    eng = BacktestEngine(_make_config([]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _make_synthetic_candles(n=5)})
    _run(eng)

    out_dir = eng.write_reports(out_dir=tmp_path / "reports")
    for fname in ("trades.csv", "equity_curve.csv", "metrics.json"):
        assert (out_dir / fname).exists(), fname
    assert "id,strategy_name" in (out_dir / "trades.csv").read_text(encoding="utf-8")


# ---- 다중 전략 배타 슬롯 경합 정책 ----


class _FirstLongStrategy(StubStrategy):
    """첫 호출에만 LONG, 이후 HOLD."""

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

    def compute_stop_loss(self, ctx, s):
        return ctx.current_price * 0.995

    def compute_take_profit(self, ctx, s, sl):
        return ctx.current_price * 1.10


class _AlwaysLongStrategy(_FirstLongStrategy):
    """슬롯이 비기만 하면 매번 LONG."""

    name = "always_long"

    def generate_signal(self, ctx):
        self.signal_calls += 1
        return Signal(side=SignalSide.LONG)


def test_multi_strategy_priority_first_wins():
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)
    candles = _make_synthetic_candles(n=40, drift=-50)
    eng = BacktestEngine(
        _make_config(["first_long", "always_long"]),
        candles.index[0].to_pydatetime(),
        candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    _run(eng)

    by_strategy: dict[str, list] = {}
    for t in eng.trades:
        by_strategy.setdefault(t["strategy_name"], []).append(t)

    assert len(eng.trades) >= 2
    assert len(by_strategy["first_long"]) == 1
    assert len(by_strategy.get("always_long", [])) >= 1
    first_entry = by_strategy["first_long"][0]["entry_time"]
    earliest_always = min(t["entry_time"] for t in by_strategy["always_long"])
    assert first_entry < earliest_always


def test_multi_strategy_lower_priority_skipped_when_slot_full():
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)
    candles = _make_synthetic_candles(n=40, drift=-50)
    eng = BacktestEngine(
        _make_config(["first_long", "always_long"]),
        candles.index[0].to_pydatetime(),
        candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    _run(eng)

    first_inst = next(s for s in eng.strategies if s.name == "first_long")
    always_inst = next(s for s in eng.strategies if s.name == "always_long")
    assert first_inst.signal_calls > 0
    assert 0 < always_inst.signal_calls < first_inst.signal_calls


def test_multi_strategy_swapped_priority():
    register_strategy(_FirstLongStrategy)
    register_strategy(_AlwaysLongStrategy)
    candles = _make_synthetic_candles(n=40, drift=-50)
    eng = BacktestEngine(
        _make_config(["always_long", "first_long"]),  # 순서 뒤집음
        candles.index[0].to_pydatetime(),
        candles.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"1m": candles})
    _run(eng)

    strategies_traded = {t["strategy_name"] for t in eng.trades}
    assert "always_long" in strategies_traded
    assert "first_long" not in strategies_traded


# ---- warmup 캔들 자동 로드 ----


def test_load_candles_includes_warmup(tmp_path, monkeypatch):
    """_load_candles 가 history_bars 만큼 start_ms 를 앞당겨 warmup 을 로드한다."""
    config = _make_config([])
    config["data"] = {"history_bars": 300, "candle_dir": str(tmp_path / "candles")}
    eng = BacktestEngine(config, "2024-06-01", "2024-06-02")
    eng.timeframes = ["15m", "4h"]

    captured: list[tuple[str, int, int]] = []

    def fake_download_range_merged(self, tf, start_ms, end_ms):
        captured.append((tf, start_ms, end_ms))
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    from src.data import historical as hist_mod
    monkeypatch.setattr(
        hist_mod.HistoricalDataLoader,
        "download_range_merged",
        fake_download_range_merged,
    )
    monkeypatch.setattr(hist_mod.HistoricalDataLoader, "close", lambda self: None)

    eng._load_candles()

    expected_start_ms = int(
        datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp() * 1000
    )
    expected_end_ms = int(
        datetime(2024, 6, 2, tzinfo=timezone.utc).timestamp() * 1000
    )
    by_tf = {tf: (s, e) for (tf, s, e) in captured}
    assert set(by_tf.keys()) == {"15m", "4h"}
    assert by_tf["15m"][0] == expected_start_ms - 300 * hist_mod.TF_MS["15m"]
    assert by_tf["15m"][1] == expected_end_ms
    assert by_tf["4h"][0] == expected_start_ms - 300 * hist_mod.TF_MS["4h"]


def test_run_loop_slices_to_original_range_after_warmup():
    """warmup 캔들이 candles_per_tf 에 포함돼도 run 루프는 [start, end]만 순회."""
    register_strategy(_TPTakerStrategy)
    config = _make_config(["tp_taker"])
    candles = _make_synthetic_candles(n=200, drift=+50)
    oos_start = candles.index[100].to_pydatetime()
    oos_end = candles.index[-1].to_pydatetime()

    eng = BacktestEngine(config, oos_start, oos_end)
    eng.inject_candles({"1m": candles})  # warmup 포함 전체 200봉
    _run(eng)

    assert len(eng.trades) >= 1
    entry_time = eng.trades[0]["entry_time"]
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time)
    assert entry_time >= oos_start
