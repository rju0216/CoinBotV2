"""룰베이스 추세추종 전략: Donchian 돌파 + Donchian exit channel trailing + 옵션 필터.

PATH_E. 단기 방향 *예측*(PATH_D NO-GO) 대신 추세에 *반응* + trailing 으로
평균 R 확대(비대칭 보상). 방향을 맞히는 게 아니라 큰 추세를 따라가 꼬리에서 번다.

진입:
  - close > 직전 entry_period 봉 최고 (Donchian upper) → LONG
  - close < 직전 entry_period 봉 최저 (Donchian lower) → SHORT
  - compute_donchian 이 shift(1)로 현재봉 제외 → 자기참조 lookahead 차단

옵션 필터 (진입 전, MS-4 정식 운영값 = LO+ma200):
  - long_only: bool — short 신호 무시 (BTC 상승편향, short PF 0.96)
  - regime_filter_type: adx/er/chop/ma/vol/di/none — 레짐 통과 시만 진입
  - regime_threshold: 레짐 임계 (방향성 지표 ma/di 는 무관)
  - ma_period: ma 필터 EMA 기간 (default 200)
  - vol_lookback / vol_max_quantile: vol 레짐 rolling 분위 (lookahead 방지)
  레짐 지표 NaN·데이터부족 시 진입 보류(HOLD, 보수적).
  옵션 off(long_only=false, regime=none) = baseline(TF-1~4 GO, PF 1.31).

청산:
  - 초기 SL = 진입가 ± ATR × atr_sl_mult
  - trailing = exit_period 봉 Donchian 반대 채널(Turtle exit). 단조 갱신(plugin 보장).
  - TP 미설정(compute_take_profit→None) → trailing SL 만 청산 → 큰 추세 다 먹음(I-PE003)

config 예시 (MS-4 채택 LO+ma200):
  strategies:
    active: ["trend_donchian"]
  trend_donchian:
    risk_per_trade_pct: 0.01
    max_leverage: 5
    entry_period: 20
    exit_period: 10
    atr_period: 14
    atr_sl_mult: 2.0
    long_only: true
    regime_filter_type: "ma"
    ma_period: 200
"""

from __future__ import annotations

import pandas as pd

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.indicators import (
    compute_adx,
    compute_atr,
    compute_choppiness,
    compute_donchian,
    compute_efficiency_ratio,
    compute_ema,
)
from src.strategy.registry import register_strategy


@register_strategy
class TrendDonchian(StrategyModule):
    name = "trend_donchian"
    entry_timeframe = "4h"
    required_timeframes = ["4h"]

    def _df(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.candles.get(self.entry_timeframe, pd.DataFrame())

    def _regime_ok(self, df: pd.DataFrame, side: SignalSide) -> bool:
        """레짐 필터 통과 여부. NaN/데이터부족 시 False(진입 보류, 보수적)."""
        rtype = str(self.params.get("regime_filter_type", "none")).lower()
        if rtype == "none":
            return True
        thr = float(self.params.get("regime_threshold", 25))
        try:
            if rtype == "adx":
                v = compute_adx(df, 14)["adx"].iloc[-1]
                return bool(v > thr) if pd.notna(v) else False
            if rtype == "er":
                v = compute_efficiency_ratio(df, 10).iloc[-1]
                return bool(v > thr) if pd.notna(v) else False
            if rtype == "chop":
                v = compute_choppiness(df, 14).iloc[-1]
                return bool(v < thr) if pd.notna(v) else False
            if rtype == "ma":
                ma_period = int(self.params.get("ma_period", 200))
                ema = compute_ema(df, ma_period)
                e = ema.iloc[-1] if ema is not None else None
                if e is None or pd.isna(e):
                    return False
                c = float(df["close"].iloc[-1])
                return c > e if side == SignalSide.LONG else c < e
            if rtype == "vol":
                lb = int(self.params.get("vol_lookback", 100))
                q = float(self.params.get("vol_max_quantile", 0.8))
                atr_pct = compute_atr(df, 14) / df["close"]
                if atr_pct.iloc[:-1].notna().sum() < lb:
                    return False
                cur = atr_pct.iloc[-1]
                # 과거 lb봉(현재 제외)의 분위 → lookahead 없음. 고변동 회피.
                thr_q = atr_pct.iloc[-lb - 1:-1].quantile(q)
                return bool(pd.notna(cur) and pd.notna(thr_q) and cur <= thr_q)
            if rtype == "di":
                adx = compute_adx(df, 14)
                p, m = adx["plus_di"].iloc[-1], adx["minus_di"].iloc[-1]
                if pd.isna(p) or pd.isna(m):
                    return False
                return p > m if side == SignalSide.LONG else m > p
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        return True

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

        meta = {"donchian_upper": float(upper), "donchian_lower": float(lower)}
        if close > float(upper):
            side = SignalSide.LONG
        elif close < float(lower):
            side = SignalSide.SHORT
        else:
            return Signal(side=SignalSide.HOLD, meta=meta)

        if bool(self.params.get("long_only", False)) and side == SignalSide.SHORT:
            return Signal(side=SignalSide.HOLD, meta={**meta, "filtered": "long_only"})
        if not self._regime_ok(df, side):
            return Signal(side=SignalSide.HOLD, meta={**meta, "filtered": "regime"})
        return Signal(side=side, meta=meta)

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
    ) -> float | None:
        # 추세추종: TP 미설정(None) → trailing SL(update_stop_loss) 만으로 청산.
        # 거래소 TP conditional order 미등록 → 큰 추세를 끝까지 보존(I-PE003).
        return None

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
