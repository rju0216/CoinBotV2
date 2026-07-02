"""RegimeQuantStrategy — BTC 1h HMM 레짐 퀀트 디스패처 플러그인 (D4-3 S2).

발견(자유)→매핑(번역)→매매(규칙) 3층 중 매매층의 진입점. 단일 StrategyModule 이
RegimeService(발견+매핑)가 낸 Contract 를 받아 TrendLogic/RangeLogic 에 위임한다
(D2 Step1 = 단일 디스패처). 레짐은 상호배타라 단일 슬롯을 자연 점유한다.

엔진 계약 매핑 (engine_base 호출 흐름):
  - generate_signal(flat)  → 현재 contract 로 trend/range 진입 신호. range 상태기계는
                             여기서 매 봉 advance (flat 일 때만 호출되므로 단일슬롯 §5.2 정합).
  - compute_stop_loss      → logic.initial_stop_loss (진입가=ctx.current_price, ATR=진입 contract).
  - compute_take_profit    → trend None / range BB 중심선.
  - compute_position_size  → logic.position_size (risk_based_size × confidence, O-2).
  - allow_entry            → True (confidence 게이트는 generate_signal).
  - update_stop_loss       → trend 트레일링 / range None(고정).
  - update_take_profit     → range 이동 중심선 / trend None.
  - should_force_exit      → 적대 전환 청산(REGIME_EXIT).
  - on_position_opened     → contract·logic 을 position.meta 에 stash (signal 미수신 보완, D2 §3).
  - on_position_closed     → range 상태기계 reset.

포지션 소속 로직은 position.meta["logic"] 로 식별 → 보유 중 hook 디스패치.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.enums import SignalSide
from src.core.types import ExitDecision, Position, Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.regime.artifact import RegimeModel, load_model
from src.strategy.regime.range_logic import RangeLogic
from src.strategy.regime.service import RegimeService
from src.strategy.regime.trend_logic import TrendLogic
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class RegimeQuantStrategy(StrategyModule):
    name = "regime_quant"
    entry_timeframe = "1h"
    required_timeframes = ["1h"]
    # range 만 SL·TP 동시 보유 → 동시 도달 시 보수적 SL 우선 (추세는 TP 없어 무의미).
    sl_tp_fill_priority = "sl_first"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.trend = TrendLogic(self.params)
        self.range = RangeLogic(self.params)
        self.service = self._build_service(self.params)
        # generate_signal → on_position_opened 사이 contract·logic 전달용 stash.
        self._pending_meta: dict[str, Any] | None = None
        # ---- 추세 진입 churn 가드 (I-010, DD-1 재검토) ----
        # cooldown_bars: 청산 후 재진입 금지. 청산봉의 on_bar_close 가 1을 소비하므로
        #   실효 대기 = N-1봉 (0=관대진입).
        # entry_on_transition_only: 전환봉(non-trend→trend)만 진입 (True=스펙§3.1 원안).
        self.cooldown_bars = int(self.params.get("cooldown_bars", 0))
        self.entry_on_transition_only = bool(
            self.params.get("entry_on_transition_only", False)
        )
        # 가드 상태 (on_bar_close 가 매 봉 갱신). None=아직 청산 이력 없음(쿨다운 비활성).
        self._bars_since_exit: int | None = None
        self._prev_trend = False  # 직전 봉이 trend 였나 (전환 판정)
        self._is_transition = False  # 이번 봉이 non-trend→trend 전환봉인가

    # ---- artifact 로딩 ----

    @staticmethod
    def _load_models(model_dir: str | Path) -> list[RegimeModel]:
        p = Path(model_dir)
        if not p.exists():
            return []
        return [load_model(f) for f in sorted(p.glob("*.json"))]

    def _build_service(self, params: dict[str, Any]) -> RegimeService:
        model_dir = params.get("model_dir", "data/regime_models")
        models = self._load_models(model_dir)
        if not models:
            raise FileNotFoundError(
                f"RegimeQuantStrategy: '{model_dir}' 에 walk-forward artifact(*.json)가 "
                "없다. 레짐 전략은 사전학습 모델이 필수 — training 파이프라인으로 먼저 "
                "artifact 를 생성하라."
            )
        return RegimeService(
            models,
            window=int(params.get("filter_window", 100)),
            atr_period=int(params.get("atr_period", 24)),
        )

    def _contract(self, ctx: StrategyContext):
        df = ctx.candles.get(self.entry_timeframe)
        if df is None or len(df) == 0:
            return None
        return self.service.get_contract(df)

    # ---- 진입 (flat 일 때만 엔진이 호출) ----

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        # churn 가드 ①: 청산 후 cooldown_bars 봉 동안 재진입 금지.
        if (
            self._bars_since_exit is not None
            and self._bars_since_exit < self.cooldown_bars
        ):
            return Signal(side=SignalSide.HOLD)

        contract = self._contract(ctx)
        df = ctx.candles.get(self.entry_timeframe)
        if contract is None or df is None or len(df) == 0:
            return Signal(side=SignalSide.HOLD)

        # range 상태기계는 매 봉 advance (trend 면 내부에서 reset+None → stale 방지).
        range_sig = self.range.advance(contract, df)

        if contract.is_trend:
            # churn 가드 ②: entry_on_transition_only 면 전환봉만 진입
            # (지속 trend 중 손절 후 재진입 = 휩쏘 차단).
            if self.entry_on_transition_only and not self._is_transition:
                sig = None
            else:
                sig = self.trend.entry_signal(contract)
            logic_name = "trend"
        elif range_sig is not None:
            sig = range_sig
            logic_name = "range"
        else:
            sig = None

        if sig is None:
            return Signal(side=SignalSide.HOLD)

        sig.meta["logic"] = logic_name
        self._pending_meta = {
            "logic": logic_name,
            "entry_contract": {
                "type": contract.type.value,
                "direction": contract.direction.value,
                "confidence": contract.confidence,
                "volatility": contract.volatility,
            },
        }
        return sig

    def compute_stop_loss(
        self, ctx: StrategyContext, signal: Signal
    ) -> float | None:
        vol = float(signal.meta.get("volatility", 0.0))
        if signal.meta.get("logic") == "trend":
            return self.trend.initial_stop_loss(signal, ctx.current_price, vol)
        return self.range.initial_stop_loss(signal, ctx.current_price, vol)

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float | None
    ) -> float | None:
        if signal.meta.get("logic") == "trend":
            return self.trend.take_profit()  # 추세 TP 없음 → None
        df = ctx.candles.get(self.entry_timeframe)
        return None if df is None else self.range.take_profit(df)

    def compute_position_size(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float | None
    ) -> float:
        if stop_loss is None:
            return 0.0  # SL 거리 기반 사이징 불가 (우리 로직은 항상 float SL)
        if signal.meta.get("logic") == "trend":
            return self.trend.position_size(
                signal, ctx.current_price, stop_loss, ctx.balance
            )
        return self.range.position_size(
            signal, ctx.current_price, stop_loss, ctx.balance
        )

    def allow_entry(self, ctx: StrategyContext) -> bool:
        return True  # confidence 게이트는 generate_signal 에 있음

    def on_bar_close(self, ctx: StrategyContext, timeframe: str) -> None:
        """매 봉(보유 무관) churn 가드 상태 갱신 — 쿨다운 카운트 + 전환 판정.

        generate_signal(flat 일 때만 호출)로는 보유 중 레짐 변화를 못 보므로,
        직전 봉 trend 여부(_prev_trend)를 여기서 매 봉 추적해 전환(non-trend→trend)을
        판정한다. entry_tf 봉에서만. (evaluate 흐름상 generate_signal 보다 먼저 호출)
        """
        if timeframe != self.entry_timeframe:
            return
        if self._bars_since_exit is not None:
            self._bars_since_exit += 1
        contract = self._contract(ctx)
        cur_trend = contract.is_trend if contract is not None else False
        self._is_transition = (not self._prev_trend) and cur_trend
        self._prev_trend = cur_trend

    # ---- 보유 중 (매 봉) ----

    def update_stop_loss(
        self, ctx: StrategyContext, position: Position
    ) -> float | None:
        if position.meta.get("logic") != "trend":
            return None  # range 는 SL 고정 (트레일링 없음)
        contract = self._contract(ctx)
        if contract is None:
            return None
        return self.trend.trailing_stop(contract, position, ctx.current_price)

    def update_take_profit(
        self, ctx: StrategyContext, position: Position
    ) -> float | None:
        if position.meta.get("logic") != "range":
            return None  # trend 는 TP 없음
        contract = self._contract(ctx)
        df = ctx.candles.get(self.entry_timeframe)
        if contract is None or df is None:
            return None
        return self.range.update_take_profit(contract, position, df)

    def should_force_exit(
        self, ctx: StrategyContext, position: Position
    ) -> ExitDecision | None:
        contract = self._contract(ctx)
        if contract is None:
            return None
        if position.meta.get("logic") == "trend":
            return self.trend.force_exit(contract, position)
        return self.range.force_exit(contract, position)

    # ---- 포지션 라이프사이클 ----

    def on_position_opened(self, position: Position) -> None:
        # 엔진은 signal.meta 를 position.meta 로 자동복사하지 않음 → 여기서 수동 기입.
        if self._pending_meta is not None:
            position.meta.update(self._pending_meta)
            self._pending_meta = None
        else:  # 방어: stash 없이 열린 경우 (정상 흐름에선 미발생)
            position.meta.setdefault("logic", "trend")
        self.range.reset()  # 진입 성공 → 보유 중 셋업 추적 불필요

    def on_position_closed(self, position: Position, pnl: float) -> None:
        self.range.reset()  # 청산 → 다음 flat 에서 fresh 셋업
        self._bars_since_exit = 0  # churn 쿨다운 시작 (on_bar_close 가 봉마다 +1)
