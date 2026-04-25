"""샘플 전략: 15m EMA 크로스.

뼈대 프로토타입의 플러그인 구조 검증용. 최소한의 진입 규칙으로
실 운영 신호 품질은 보장하지 않음.

진입:
  - 빠른 EMA가 느린 EMA를 상향 크로스 → LONG
  - 빠른 EMA가 느린 EMA를 하향 크로스 → SHORT
  - 그 외 HOLD

SL = 진입가 ± ATR × atr_sl_mult
TP = 진입가 ± |진입가 - SL| × reward_risk_ratio

config 예시:
  strategies:
    active: ["example_macross"]
  example_macross:
    risk_per_trade_pct: 0.01       # 필수 (엔진 사이징)
    max_leverage: 5                # 필수 (엔진 사이징)
    ma_fast: 20
    ma_slow: 50
    atr_period: 14
    atr_sl_mult: 1.5
    reward_risk_ratio: 2.0
"""

from __future__ import annotations

import pandas as pd

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.indicators import compute_atr, compute_ema
from src.strategy.registry import register_strategy


@register_strategy
class ExampleMACross(StrategyModule):
    name = "example_macross"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def _df(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.candles.get(self.entry_timeframe, pd.DataFrame())

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        df = self._df(ctx)
        ma_fast_p = int(self.params.get("ma_fast", 20))
        ma_slow_p = int(self.params.get("ma_slow", 50))
        if len(df) < ma_slow_p + 2:
            return Signal(side=SignalSide.HOLD)

        ma_fast = compute_ema(df, ma_fast_p)
        ma_slow = compute_ema(df, ma_slow_p)
        prev_diff = float(ma_fast.iloc[-2] - ma_slow.iloc[-2])
        curr_diff = float(ma_fast.iloc[-1] - ma_slow.iloc[-1])

        meta = {
            "ma_fast": float(ma_fast.iloc[-1]),
            "ma_slow": float(ma_slow.iloc[-1]),
        }
        if prev_diff <= 0 and curr_diff > 0:
            return Signal(side=SignalSide.LONG, meta=meta)
        if prev_diff >= 0 and curr_diff < 0:
            return Signal(side=SignalSide.SHORT, meta=meta)
        return Signal(side=SignalSide.HOLD, meta=meta)

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        df = self._df(ctx)
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_sl_mult", 1.5))
        if len(df) < atr_period + 1:
            # 폴백: 진입가 ±0.5%
            return (
                ctx.current_price * 0.995
                if signal.side == SignalSide.LONG
                else ctx.current_price * 1.005
            )
        atr_value = float(compute_atr(df, atr_period).iloc[-1])
        if signal.side == SignalSide.LONG:
            return ctx.current_price - atr_value * atr_mult
        return ctx.current_price + atr_value * atr_mult

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        rr = float(self.params.get("reward_risk_ratio", 2.0))
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr
