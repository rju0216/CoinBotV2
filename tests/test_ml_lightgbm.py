"""Phase B-1a: LightGBM 전략 플러그인 단위 테스트.

플러그인 등록, config 파라미터 처리, SL/TP 계산을 검증.
모델 추론 테스트는 lightgbm 설치 + 학습된 모델이 필요하므로 별도 분리.
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


class TestMLLightGBMRegistration:
    def test_plugin_registered(self):
        """@register_strategy 데코레이터로 ml_lightgbm이 등록됨."""
        from src.strategy.plugins.ml_lightgbm import MLLightGBM
        from src.strategy.registry import register_strategy

        # reset 후 재등록
        register_strategy(MLLightGBM)
        strategy_cls = get_strategy_class("ml_lightgbm")
        assert strategy_cls is MLLightGBM
        assert strategy_cls.name == "ml_lightgbm"

    def test_default_timeframe(self):
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        assert s.entry_timeframe == "15m"
        assert "15m" in s.required_timeframes

    def test_config_timeframe_override(self):
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "entry_timeframe": "1h",
            "required_timeframes": ["1h", "4h"],
        })
        assert s.entry_timeframe == "1h"
        assert s.required_timeframes == ["1h", "4h"]

    def test_entry_tf_auto_included(self):
        """entry_timeframe이 required_timeframes에 없으면 자동 추가."""
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "entry_timeframe": "1h",
            "required_timeframes": ["4h"],
        })
        assert "1h" in s.required_timeframes


class TestMLLightGBMSLTP:
    def test_stop_loss_long(self):
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_period": 14,
            "atr_sl_mult": 2.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, signal)
        assert sl < 67000.0

    def test_stop_loss_short(self):
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_period": 14,
            "atr_sl_mult": 2.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.SHORT)
        sl = s.compute_stop_loss(ctx, signal)
        assert sl > 67000.0

    def test_take_profit_long(self):
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "atr_sl_mult": 2.0,
            "reward_risk_ratio": 2.0,
        })
        ctx = _make_ctx(current_price=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, signal)
        tp = s.compute_take_profit(ctx, signal, sl)
        assert tp > 67000.0
        # TP 거리는 SL 거리의 reward_risk_ratio 배
        risk = abs(67000.0 - sl)
        reward = abs(tp - 67000.0)
        assert abs(reward / risk - 2.0) < 0.01

    def test_stop_loss_fallback_short_data(self):
        """데이터가 ATR 기간보다 짧으면 폴백 (±0.5%)."""
        from src.strategy.plugins.ml_lightgbm import MLLightGBM

        s = MLLightGBM({
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
        assert sl > 67000.0 * 0.99  # 폴백은 ±0.5%


class TestTrainScriptImport:
    def test_script_importable(self):
        """train_lightgbm.py가 import 가능한지 확인 (구문 오류 검증)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_lightgbm",
            "scripts/train_lightgbm.py",
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        # 실행하지 않고 로드만 시도 — lightgbm 미설치 시 ImportError
        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main")
        except ImportError:
            pytest.skip("lightgbm 미설치")
