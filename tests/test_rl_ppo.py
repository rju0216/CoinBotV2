"""Phase B-3: PPO 전략 플러그인 단위 테스트.

플러그인 등록, config 파라미터 처리(lookback 포함), SL/TP 계산을 검증.
모델 추론 + should_force_exit 테스트는 학습된 PPO + scaler가 필요하므로 별도 분리.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.registry import get_strategy_class, reset_registry_for_testing


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _make_candles(n: int = 300, start_price: float = 67000.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(10, 100, n)
    low = close - rng.uniform(10, 100, n)
    open_ = close + rng.normal(0, 30, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_ctx(
    candles: dict[str, pd.DataFrame] | None = None,
    current_price: float = 67000.0,
) -> StrategyContext:
    if candles is None:
        candles = {"15m": _make_candles(300)}
    return StrategyContext(
        candles=candles,
        current_price=current_price,
        balance=10000.0,
        position=None,
        is_slot_occupied=False,
        params={},
        now=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


class TestRLPPORegistration:
    def test_plugin_registered(self):
        from src.strategy.plugins.rl_ppo import RLPPO
        from src.strategy.registry import register_strategy

        register_strategy(RLPPO)
        strategy_cls = get_strategy_class("rl_ppo")
        assert strategy_cls is RLPPO
        assert strategy_cls.name == "rl_ppo"

    def test_default_timeframe(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        assert s.entry_timeframe == "15m"
        assert "15m" in s.required_timeframes

    def test_config_timeframe_override(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "entry_timeframe": "1h",
            "required_timeframes": ["1h", "4h"],
        })
        assert s.entry_timeframe == "1h"
        assert s.required_timeframes == ["1h", "4h"]

    def test_entry_tf_auto_included(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "entry_timeframe": "1h",
            "required_timeframes": ["4h"],
        })
        assert "1h" in s.required_timeframes

    def test_lookback_default(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        assert s._lookback == 60

    def test_lookback_override(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "lookback": 30,
        })
        assert s._lookback == 30


class TestRLPPOSLTP:
    """RL은 SL/TP를 안전장치로만 사용 — atr_sl_mult=5.0, RR=10.0 (매우 넓게)."""

    def test_stop_loss_long(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_period": 14,
            "atr_sl_mult": 5.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, signal)
        assert sl < 67000.0

    def test_stop_loss_short(self):
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_period": 14,
            "atr_sl_mult": 5.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.SHORT)
        sl = s.compute_stop_loss(ctx, signal)
        assert sl > 67000.0

    def test_sl_wider_than_ml(self):
        """RL의 SL 거리가 ML(atr_mult=2.0) 대비 약 2.5배 넓은지 확인."""
        from src.strategy.plugins.rl_ppo import RLPPO

        s_rl = RLPPO({
            "risk_per_trade_pct": 0.01, "max_leverage": 5,
            "atr_period": 14, "atr_sl_mult": 5.0,
        })
        s_ml = RLPPO({
            "risk_per_trade_pct": 0.01, "max_leverage": 5,
            "atr_period": 14, "atr_sl_mult": 2.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl_rl = s_rl.compute_stop_loss(ctx, signal)
        sl_ml = s_ml.compute_stop_loss(ctx, signal)
        # SL 거리 (현재가 - SL): RL이 ML의 2.5배
        ratio = (67000.0 - sl_rl) / (67000.0 - sl_ml)
        assert abs(ratio - 2.5) < 0.05

    def test_take_profit_rr_10(self):
        """RR=10.0 검증."""
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_sl_mult": 5.0,
            "reward_risk_ratio": 10.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, signal)
        tp = s.compute_take_profit(ctx, signal, sl)
        risk = abs(67000.0 - sl)
        reward = abs(tp - 67000.0)
        assert abs(reward / risk - 10.0) < 0.01

    def test_stop_loss_fallback_short_data(self):
        """데이터 부족 시 폴백 (±1.0%, ML의 0.5%보다 넓게)."""
        from src.strategy.plugins.rl_ppo import RLPPO

        s = RLPPO({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_period": 14,
        })
        ctx = _make_ctx(
            candles={"15m": _make_candles(5)},
            current_price=67000.0,
        )
        signal = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, signal)
        assert sl < 67000.0
        # 폴백은 ±1% — RL은 더 넓음
        assert sl < 67000.0 * 0.995


class TestTrainScriptImport:
    def test_script_importable(self):
        """train_ppo.py가 import 가능한지 확인."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_ppo",
            "scripts/train_ppo.py",
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main")
        except ImportError:
            pytest.skip("stable-baselines3 미설치")
