"""Phase 6 Step 6.2 — 적응형 청산 커스텀 3층 (A: insight-기반 재설계).

DumbL3 를 상속해 **진입은 그대로**(θ·방향·고변동 유지·트렌드/conviction 필터 없음) 두고
**청산만 새로 설계**한다. 6.1 소진이 준 재료:
- ② 고정 대칭배리어가 "작은이익·큰손실" payoff (2025 win<be, 98거래).
- D-034: 엣지=단기지평 → 빠른 익절(TP)은 유지, **손실 쪽**을 재설계.

**청산 레버 (전부 원리고정·무튜닝 — θ 과적합 교훈)**:
- **E1 signal_decay**: 보유 방향의 fresh p_dir < 진입θ 면 조기청산. TP·SL·만기 **유지**(만기 backstop
  → D-034 hold-past-horizon 회피). expire 로 갈 거래를 신호 꺼질 때 미리 정리.
- **E2 sl_mult**: SL 배수(0.5=손절 빠르게). 바로 내리꽂는 손실을 자름.
- **E4 breakeven**: 초기 SL 유지 → 이익 breakeven_r×R 도달 시 SL 을 진입가로. 올랐다 반전하는
  손실을 휘프소 없이 차단(초기 SL 안 조임). E2 와 다른 손실 인구 겨냥(상호보완).

플래그 조합으로 개별/결합 표현. 엔진 무손(훅만). 아티팩트 열 요구는 DumbL3 와 동일.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import ExitDecision, Position, Signal, StrategyContext
from src.data.historical import TF_MS
from src.strategy.plugins.dumb_l3 import DumbL3
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class AdaptiveL3(DumbL3):
    """적응 청산 3층 — DumbL3 진입 재사용 + 청산 재설계(E1/E2/E4)."""

    name = "adaptive_l3"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # 청산 레버 (원리고정). 기본값 = dumb 와 동일 동작(전부 off/1.0).
        self.signal_decay = bool(self.params.get("signal_decay", False))   # E1
        self.sl_mult = float(self.params.get("sl_mult", 1.0))              # E2 (0.5=tight)
        self.breakeven = bool(self.params.get("breakeven", False))          # E4
        self.breakeven_r = float(self.params.get("breakeven_r", 1.0))       # 이익 R배 도달시 BE
        # per-position 상태 (단일슬롯 — 한 번에 한 포지션)
        self._be_trigger: float | None = None
        self._be_armed: bool = False
        self._decay_key = None       # 신호감쇠 lookup 캐시(4h봉 단위, 1m 반복 방지)
        self._decay_exit: bool = False
        logger.info("AdaptiveL3: signal_decay=%s sl_mult=%.2f breakeven=%s(r=%.1f)",
                    self.signal_decay, self.sl_mult, self.breakeven, self.breakeven_r)

    # ---- E2: SL 배수 (진입 SL 만 조정, TP·방향 로직은 dumb 그대로) ----
    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float | None:
        w = signal.meta.get("barrier_frac")
        if w is None:
            return None
        w = w * self.sl_mult
        entry = ctx.current_price
        return entry * (1.0 - w) if signal.side == SignalSide.LONG else entry * (1.0 + w)

    # ---- E4: breakeven 상태 초기화 (진입 시 1R 트리거 가격 확정) ----
    def on_position_opened(self, position: Position) -> None:
        self._be_armed = False
        self._be_trigger = None
        self._decay_key = None
        self._decay_exit = False
        if self.breakeven and position.stop_loss is not None:
            r = abs(position.entry_price - position.stop_loss)   # 초기 리스크 R (=SL 거리)
            if position.side == PositionSide.LONG:
                self._be_trigger = position.entry_price + self.breakeven_r * r
            else:
                self._be_trigger = position.entry_price - self.breakeven_r * r

    # ---- E4: 이익 R배 도달 → SL 을 진입가로 (한 번만) ----
    def update_stop_loss(self, ctx: StrategyContext, position: Position) -> float | None:
        if not self.breakeven or self._be_armed or self._be_trigger is None:
            return None
        px = ctx.current_price
        hit = (px >= self._be_trigger if position.side == PositionSide.LONG
               else px <= self._be_trigger)
        if hit:
            self._be_armed = True
            return position.entry_price       # SL → breakeven
        return None

    # ---- E1: 만기 backstop(super) + 신호감쇠 조기청산 ----
    def should_force_exit(self, ctx: StrategyContext, position: Position) -> ExitDecision | None:
        base = super().should_force_exit(ctx, position)   # 만기(timeout) 유지
        if base is not None:
            return base
        if not self.signal_decay:
            return None
        df = ctx.candles.get(self.entry_timeframe)
        if df is None or len(df) == 0:
            return None
        # ★완성봉 가드 (lookahead 방지)★: 1m fill 하에서 should_force_exit 는 매 1m 호출되고
        # _slice_candles(index<ts)는 **진행 중** 결정TF봉(open<ts, 마감>ts)을 포함한다. 그 봉의
        # 예측(=봉 마감이 만든 값)을 쓰면 미래 누수 → **마감<=now 인 완성봉만** 사용.
        now = pd.Timestamp(ctx.now)
        idx = df.index
        if now.tzinfo is None and idx.tz is not None:
            now = now.tz_localize(idx.tz)
        elif now.tzinfo is not None and idx.tz is None:
            now = now.tz_localize(None)
        interval = pd.Timedelta(milliseconds=TF_MS[self.entry_timeframe])
        completed = idx[idx + interval <= now]             # 마감시각 <= 현재 인 봉만
        if len(completed) == 0:
            return None
        key = completed[-1]                                # 마지막 **완성** 결정TF 봉(인과)
        if key != self._decay_key:                         # 4h봉 바뀔 때만 재평가(1m 반복 캐시)
            self._decay_key = key
            row = self._lookup(key)
            if row is None:
                self._decay_exit = False
            else:
                s = self.pred_source
                held_p = (float(row[f"{s}_up"]) if position.side == PositionSide.LONG
                          else float(row[f"{s}_down"]))
                self._decay_exit = held_p < self.theta     # 보유 방향 신뢰도가 진입기준 밑
        if self._decay_exit:
            return ExitDecision(reason=ExitReason.FORCE_EXIT, note="signal_decay")
        return None
