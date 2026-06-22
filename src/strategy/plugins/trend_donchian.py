"""룰베이스 추세추종 전략: Donchian 돌파 + Donchian exit channel trailing.

PATH_E TF-1. 단기 방향 *예측*(PATH_D NO-GO) 대신 추세에 *반응* + trailing 으로
평균 R 확대(비대칭 보상). 방향을 맞히는 게 아니라 큰 추세를 따라가 꼬리에서 번다.

진입:
  - close > 직전 entry_period 봉 최고 (Donchian upper) → LONG
  - close < 직전 entry_period 봉 최저 (Donchian lower) → SHORT
  - compute_donchian 이 shift(1)로 현재봉 제외 → 자기참조 lookahead 차단

청산:
  - 초기 SL = 진입가 ± ATR × atr_sl_mult
  - trailing = exit_period 봉 Donchian 반대 채널(Turtle exit). 단조 갱신(plugin 보장).
  - TP 사실상 비활성(reward_risk_ratio 크게) → trailing 만 청산 → 큰 추세 다 먹음

config 예시:
  strategies:
    active: ["trend_donchian"]
  trend_donchian:
    risk_per_trade_pct: 0.01
    max_leverage: 5
    entry_period: 20
    exit_period: 10
    atr_period: 14
    atr_sl_mult: 2.0
    reward_risk_ratio: 100   # TP 사실상 비활성 (trailing 청산)
"""

from __future__ import annotations

import pandas as pd

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.indicators import compute_atr, compute_donchian
from src.strategy.registry import register_strategy


@register_strategy
class TrendDonchian(StrategyModule):
    name = "trend_donchian"
    entry_timeframe = "4h"
    required_timeframes = ["4h"]

    def _df(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.candles.get(self.entry_timeframe, pd.DataFrame())

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        df = self._df(ctx)
        entry_period = int(self.params.get("entry_period", 20))
        if len(df) < entry_period + 2:
            return Signal(side=SignalSide.HOLD)

        ch = compute_donchian(df, entry_period)
        upper, lower = ch["upper"].iloc[-1], ch["lower"].iloc[-1]
        close = float(df["close"].iloc[-1])
        if pd.isna(upper) or pd.isna(lower):
            return Signal(side=SignalSide.HOLD)

        meta = {
            "donchian_upper": float(upper),
            "donchian_lower": float(lower),
            "close": close,
        }
        if close > float(upper):
            return Signal(side=SignalSide.LONG, meta=meta)
        if close < float(lower):
            return Signal(side=SignalSide.SHORT, meta=meta)
        return Signal(side=SignalSide.HOLD, meta=meta)

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        df = self._df(ctx)
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_sl_mult", 2.0))
        if len(df) < atr_period + 1:
            offset = ctx.current_price * 0.01
            return (
                ctx.current_price - offset
                if signal.side == SignalSide.LONG
                else ctx.current_price + offset
            )
        atr_val = float(compute_atr(df, atr_period).iloc[-1])
        if signal.side == SignalSide.LONG:
            return ctx.current_price - atr_val * atr_mult
        return ctx.current_price + atr_val * atr_mult

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        # 추세추종: TP 사실상 비활성(큰 RR) → trailing 으로 청산해 큰 추세 보존
        rr = float(self.params.get("reward_risk_ratio", 100.0))
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr

    def update_stop_loss(
        self, ctx: StrategyContext, position: Position
    ) -> float | None:
        """exit_period Donchian 반대 채널로 trailing. 단조 갱신은 plugin 이 보장.

        엔진(check_strategy_exits)은 반환값을 position.stop_loss 에 단순 대입하므로,
        SL 이 불리하게 후퇴하지 않도록 LONG=max / SHORT=min 으로 단조성을 보장한다.
        """
        df = self._df(ctx)
        exit_period = int(self.params.get("exit_period", 10))
        if len(df) < exit_period + 2:
            return None
        ch = compute_donchian(df, exit_period)
        cur_sl = position.stop_loss

        if position.side == PositionSide.LONG:
            new_sl = ch["lower"].iloc[-1]   # 직전 exit_period 봉 최저
            if pd.isna(new_sl):
                return None
            new_sl = float(new_sl)
            return new_sl if cur_sl is None else max(cur_sl, new_sl)

        new_sl = ch["upper"].iloc[-1]       # SHORT: 직전 exit_period 봉 최고
        if pd.isna(new_sl):
            return None
        new_sl = float(new_sl)
        return new_sl if cur_sl is None else min(cur_sl, new_sl)
