"""뼈대 프로토타입 공통 엔진 베이스.

CoreEngine(라이브/페이퍼)와 BacktestEngine이 공유하는 흐름:
  봉 마감 → 전략 평가 → (슬롯 빔) 진입 또는 (슬롯 참) SL/TP·강제청산 검사
  → 청산 시 PnL 정산 → DB 기록.

구체 엔진은 데이터 수신 방식과 run 루프만 오버라이드.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.accounting.fee_model import FeeModel
from src.core.enums import (
    EventType,
    ExitReason,
    PositionSide,
    PositionStatus,
    SignalSide,
)
from src.core.event_bus import EventBus
from src.core.policies import (
    IgnoreReversePolicy,
    ReverseSignalPolicy,
    build_reverse_policy,
)
from src.core.types import ExitDecision, Position, Signal, StrategyContext
from src.execution.broker import Broker
from src.risk.manager import RiskManager
from src.strategy.base import StrategyModule
from src.strategy.indicators import compute_atr
from src.strategy.registry import load_active_strategies

logger = logging.getLogger(__name__)


_TF_PRIORITY = {
    "1m": 0,
    "5m": 1,
    "15m": 2,
    "1h": 3,
    "4h": 4,
    "1d": 5,
}


def signal_side_to_position_side(side: SignalSide) -> PositionSide:
    if side == SignalSide.LONG:
        return PositionSide.LONG
    if side == SignalSide.SHORT:
        return PositionSide.SHORT
    return PositionSide.NONE


class AbstractEngine(ABC):
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        self.config = config
        self.mode = mode
        self.event_bus = EventBus()
        self.fee_model = FeeModel.from_config(config)
        self.risk_manager = RiskManager(config)
        self.broker = Broker(config, mode)

        engine_cfg = config.get("engine", {}) or {}
        self.reverse_policy: ReverseSignalPolicy = build_reverse_policy(
            engine_cfg.get("reverse_signal_policy", "ignore")
        )

        self.strategies: list[StrategyModule] = load_active_strategies(config)
        self.strategy_by_name: dict[str, StrategyModule] = {
            s.name: s for s in self.strategies
        }

        self.timeframes: list[str] = self._compute_timeframe_union()
        self.master_timeframe: str | None = (
            self.timeframes[0] if self.timeframes else None
        )

        # (C) 배타적 경합 — 전역 슬롯 1개
        self._position: Position | None = None

        # Phase E-2-2-OPT Step 1: entry_timeframe별 features 사전계산 cache.
        # BacktestEngine.initialize에서 OOS 전체로 채움. CoreEngine(라이브)에선
        # 빈 dict 유지 → _build_ctx가 None 반환 → plugin이 즉시 계산 경로 사용.
        self._features_cache: dict[str, pd.DataFrame] = {}

        # BP-2-3: Live OOS monitor (CoreEngine.initialize에서만 채움; 백테는 None 유지)
        # 순환 import 방지: 런타임 type만 Any로 두고 CoreEngine이 LiveOOSMonitor 주입
        self.oos_monitor: Any | None = None

    # ---- properties ----

    @property
    def position(self) -> Position | None:
        return self._position

    # ---- 추상: 구체 엔진이 구현 ----

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def run(self) -> None: ...

    # ---- 거래 기록 추상화 (라이브: DB / 백테: 메모리) ----

    @abstractmethod
    async def _record_trade_open(
        self,
        strategy_name: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        now: datetime,
    ) -> int:
        """진입 기록. 반환값은 trade_id."""

    @abstractmethod
    async def _record_trade_close(
        self,
        trade_id: int,
        position: Position,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        trading_fee: float,
        funding_fee: float,
        exit_reason: str,
        now: datetime,
    ) -> None:
        """청산 기록."""

    # ---- 활성 TF 산출 ----

    def _compute_timeframe_union(self) -> list[str]:
        """활성 전략의 entry_timeframe + required_timeframes 합집합.
        가장 작은 TF가 첫 번째 (마스터 루프 후보).
        """
        seen: set[str] = set()
        for s in self.strategies:
            if s.entry_timeframe:
                seen.add(s.entry_timeframe)
            for tf in s.required_timeframes:
                if tf:
                    seen.add(tf)
        return sorted(seen, key=lambda tf: _TF_PRIORITY.get(tf, 99))

    # ---- 컨텍스트 빌더 ----

    def _build_ctx(
        self,
        strategy: StrategyModule,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
    ) -> StrategyContext:
        # 슬롯이 이 전략 소유면 해당 Position 노출, 아니면 None
        own_position = (
            self._position
            if self._position is not None
            and self._position.strategy_name == strategy.name
            else None
        )
        return StrategyContext(
            candles=dict(candles_per_tf),
            current_price=current_price,
            balance=balance,
            position=own_position,
            is_slot_occupied=self._position is not None,
            params=strategy.params,
            now=now,
            precomputed_features=self._features_cache.get(strategy.entry_timeframe),
        )

    # ---- 봉 마감 dispatch ----

    async def evaluate_strategies_on_bar(
        self,
        bar_close_tf: str,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
    ) -> None:
        """봉 마감 시 전략 평가.

        1) 모든 전략에 on_bar_close 훅 dispatch (관련 TF에 한해)
        2) 슬롯이 비었으면: bar_close_tf == entry_timeframe 인 전략을 우선순위로
           generate_signal 호출, 첫 actionable 신호로 진입 시도 (C 정책)
        3) 슬롯이 차있고 reverse_policy != ignore 이면 reverse 검사
        4) 슬롯이 차있고 보유 전략이 supports_pyramiding 이면 generate_pyramid_signal
        """
        # 1) on_bar_close 훅
        for strategy in self.strategies:
            if (
                bar_close_tf == strategy.entry_timeframe
                or bar_close_tf in strategy.required_timeframes
            ):
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                strategy.on_bar_close(ctx, bar_close_tf)

        # 2) 슬롯 빔 → 진입 시도
        if self._position is None:
            for strategy in self.strategies:
                if strategy.entry_timeframe != bar_close_tf:
                    continue
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                signal = strategy.generate_signal(ctx)
                self._record_oos_signal(strategy, signal, current_price, now)
                self._log_signal_status(strategy, signal)
                if not signal.is_actionable:
                    continue
                if await self.try_enter(strategy, signal, ctx, now):
                    return  # (C) 첫 진입 성공 시 종료
            return

        # 3) 슬롯 참 + reverse_policy 적용 — ignore이면 스킵
        if not isinstance(self.reverse_policy, IgnoreReversePolicy):
            current_strategy_name = self._position.strategy_name
            for strategy in self.strategies:
                if strategy.entry_timeframe != bar_close_tf:
                    continue
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                signal = strategy.generate_signal(ctx)
                self._record_oos_signal(strategy, signal, current_price, now)
                self._log_signal_status(strategy, signal)
                if not signal.is_actionable:
                    continue
                if self.reverse_policy.should_reverse(
                    self._position,
                    signal,
                    current_strategy_name,
                    strategy.name,
                ):
                    await self.close_position(
                        current_price, ExitReason.REVERSE_SIGNAL, now=now
                    )
                    if await self.try_enter(strategy, signal, ctx, now):
                        return

        # 4) 피라미딩 (보유 전략이 opt-in 인 경우)
        held_strategy = self.strategy_by_name.get(self._position.strategy_name)
        if held_strategy is not None and held_strategy.supports_pyramiding:
            if held_strategy.entry_timeframe == bar_close_tf:
                ctx = self._build_ctx(
                    held_strategy, candles_per_tf, current_price, balance, now
                )
                pyramid_signal = held_strategy.generate_pyramid_signal(
                    ctx, self._position
                )
                if pyramid_signal is not None and pyramid_signal.is_actionable:
                    logger.info(
                        "Pyramid signal from %s (not yet implemented)",
                        held_strategy.name,
                    )
                    # 단계 8: 피라미딩 hook은 받되 실 청산/추가 진입 처리는 미구현.
                    # 향후 단계에서 add_to_position 흐름으로 확장 가능.

    # ---- BL-2-3 hotfix-E: 신호/포지션 모니터링 hook ----

    def _log_signal_status(
        self,
        strategy: StrategyModule,
        signal: Signal,
    ) -> None:
        """generate_signal 결과를 모니터링 로그로 출력하는 hook (default no-op).

        backtest는 매 봉 수만 줄 출력 회피 위해 default no-op. CoreEngine이
        override해서 라이브/페이퍼 모드에서만 INFO 출력.
        """
        return None

    def _log_position_status(
        self,
        position: Position,
        current_price: float,
        now: datetime,
    ) -> None:
        """현재 포지션 상태(side/entry/current/unrealized_pnl/hold_duration)
        모니터링 로그 hook (default no-op). 라이브/페이퍼만 활성.
        """
        return None

    def _log_account_status(
        self,
        balance: float,
        current_price: float,
    ) -> None:
        """현재 계정 재정 상태(balance/equity/unrealized/daily_pnl/dd) 모니터링
        로그 hook (default no-op). 라이브/페이퍼만 활성. master_tf 봉 마감 시
        포지션 유무 무관 호출.
        """
        return None

    # ---- BP-2-3 OOS monitor helper ----

    def _record_oos_signal(
        self,
        strategy: StrategyModule,
        signal: Signal,
        current_price: float,
        now: datetime,
    ) -> None:
        """generate_signal 결과를 oos_monitor에 push (라이브 전용; backtest는 monitor=None)."""
        if self.oos_monitor is None:
            return
        try:
            self.oos_monitor.record_prediction(
                strategy_name=strategy.name,
                entry_timeframe=strategy.entry_timeframe,
                ts=now,
                signal_side=signal.side,
                entry_close=current_price,
            )
        except Exception as e:
            logger.warning("oos_monitor.record_prediction failed: %s", e)

    # ---- BP-2-2 동적 사이징 helper ----

    def _compute_volatility_factor(
        self, strategy: StrategyModule, ctx: StrategyContext
    ) -> float:
        """volatility_targeting.enabled=true일 때 entry_timeframe ATR_pct 기반 factor.

        factor = current_atr_pct / target_atr_pct (>1 → 변동성 평소 이상 → size 축소)
        반환 1.0이면 비활성과 동일 (RiskManager가 size 변경 안 함).
        """
        vt_cfg = (self.config.get("risk", {}) or {}).get(
            "volatility_targeting", {}
        ) or {}
        if not vt_cfg.get("enabled", False):
            return 1.0
        target = float(vt_cfg.get("target_atr_pct", 0.005))
        lookback = int(vt_cfg.get("lookback", 14))
        if target <= 0:
            return 1.0
        df = ctx.candles.get(strategy.entry_timeframe)
        if df is None or len(df) < lookback + 1:
            return 1.0
        try:
            atr_series = compute_atr(df, period=lookback)
        except Exception:
            return 1.0
        if atr_series is None or atr_series.empty:
            return 1.0
        atr = float(atr_series.iloc[-1])
        close = float(df["close"].iloc[-1])
        if close <= 0 or pd.isna(atr) or atr <= 0:
            return 1.0
        current_atr_pct = atr / close
        return current_atr_pct / target

    # ---- 진입 ----

    async def try_enter(
        self,
        strategy: StrategyModule,
        signal: Signal,
        ctx: StrategyContext,
        now: datetime,
    ) -> bool:
        if self._position is not None:
            return False

        # BL-2-1: Circuit breaker OPEN 시 새 진입 차단 (사안 U''=나 trade 일시 중단)
        if getattr(self, "_circuit_breaker_open", False):
            return False

        # 위험 검증
        if not self.risk_manager.validate_order(
            ctx.balance, current_position_count=0
        ):
            logger.info("Risk validation rejected entry for %s", strategy.name)
            return False

        # SL/TP
        sl_price = float(strategy.compute_stop_loss(ctx, signal))
        tp_price = float(strategy.compute_take_profit(ctx, signal, sl_price))

        # 사이징 — 전략 params에서 risk_per_trade_pct/max_leverage 추출
        try:
            risk_pct = float(strategy.params["risk_per_trade_pct"])
            max_lev = float(strategy.params["max_leverage"])
        except KeyError as e:
            logger.error(
                "Strategy '%s' missing required param: %s",
                strategy.name,
                e,
            )
            return False

        volatility_factor = self._compute_volatility_factor(strategy, ctx)

        size = self.risk_manager.calculate_position_size(
            ctx.current_price,
            sl_price,
            ctx.balance,
            risk_per_trade_pct=risk_pct,
            max_leverage=max_lev,
            volatility_factor=volatility_factor,
        )
        if size <= 0:
            logger.info("Sizing returned 0 for %s", strategy.name)
            return False

        # 진입 주문
        position_side = signal_side_to_position_side(signal.side)
        # BL-2-2: paper 모드에서 호가창 가용 시 VWAP 침투 가격 사용 (silent fallback)
        order = await self.broker.open_position(
            position_side, size, fill_price=ctx.current_price,
            orderbook=getattr(self, "_latest_orderbook", None),
        )
        if not order:
            logger.error("Open order failed for %s", strategy.name)
            return False

        # 거래 기록 (구체 엔진이 DB 또는 메모리에 저장)
        trade_id = await self._record_trade_open(
            strategy_name=strategy.name,
            side=position_side,
            size=size,
            entry_price=ctx.current_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            now=now,
        )

        # 거래소 SL/TP pending (페이퍼는 no-op)
        await self.broker.place_stop_loss(position_side, sl_price, size)
        await self.broker.place_take_profit(position_side, tp_price, size)

        # Position 등록
        self._position = Position(
            side=position_side,
            size=size,
            entry_price=ctx.current_price,
            entry_time=now,
            strategy_name=strategy.name,
            stop_loss=sl_price,
            take_profit=tp_price,
            trade_id=trade_id,
            status=PositionStatus.OPEN,
        )

        try:
            strategy.on_position_opened(self._position)
        except Exception as e:
            logger.error(
                "on_position_opened hook error in %s: %s",
                strategy.name,
                e,
                exc_info=True,
            )

        await self.event_bus.publish(
            EventType.POSITION_OPENED.value, self._position
        )
        logger.info(
            "ENTRY[%s]: %s %.4f @ %.2f, SL=%.2f, TP=%.2f, trade_id=%d",
            strategy.name,
            position_side.value,
            size,
            ctx.current_price,
            sl_price,
            tp_price,
            trade_id,
        )
        return True

    # ---- 청산 ----

    async def close_position(
        self,
        exit_price: float,
        reason: ExitReason,
        funding_fee: float = 0.0,
        now: datetime | None = None,
    ) -> None:
        if self._position is None:
            return
        pos = self._position
        now = now or datetime.now(timezone.utc)

        # pending 주문 취소
        try:
            await self.broker.cancel_all_orders()
        except Exception as e:
            logger.warning("cancel_all_orders failed: %s", e)

        # 거래소/시뮬 청산
        # BL-2-2: paper 모드에서 호가창 가용 시 VWAP 침투 가격 사용 (silent fallback)
        await self.broker.close_position(
            pos.side, pos.size, fill_price=exit_price,
            orderbook=getattr(self, "_latest_orderbook", None),
        )

        # 수수료·PnL 정산 — FeeModel 단일 공식
        fees = self.fee_model.estimate_round_trip(
            pos.entry_price, exit_price, pos.size
        )
        pnl_result = self.fee_model.calc_pnl(
            pos.side,
            pos.entry_price,
            exit_price,
            pos.size,
            fees=fees,
            funding=funding_fee,
        )
        net_pnl = pnl_result["net_pnl"]

        # 청산 기록 (구체 엔진이 DB 또는 메모리에 저장)
        if pos.trade_id is not None:
            await self._record_trade_close(
                trade_id=pos.trade_id,
                position=pos,
                exit_price=exit_price,
                pnl=net_pnl,
                pnl_pct=pnl_result["pnl_pct"],
                trading_fee=fees,
                funding_fee=funding_fee,
                exit_reason=reason.value,
                now=now,
            )

        # 위험 매니저 갱신
        self.risk_manager.add_pnl(net_pnl)
        try:
            balance = await self.broker.get_balance()
            self.risk_manager.update_equity(balance)
        except Exception as e:
            logger.warning("get_balance failed during close: %s", e)

        # 전략 훅 (orphan이면 스킵 + 경고)
        strategy = self.strategy_by_name.get(pos.strategy_name)
        if strategy is not None:
            try:
                strategy.on_position_closed(pos, net_pnl)
            except Exception as e:
                logger.error(
                    "on_position_closed hook error in %s: %s",
                    strategy.name,
                    e,
                    exc_info=True,
                )
        else:
            logger.warning(
                "Closed orphan position (strategy '%s' not in active list)",
                pos.strategy_name,
            )

        await self.event_bus.publish(
            EventType.POSITION_CLOSED.value,
            {"position": pos, "pnl": net_pnl, "reason": reason.value},
        )
        logger.info(
            "EXIT[%s]: %s @ %.2f, reason=%s, net_pnl=$%.2f, fees=$%.2f",
            pos.strategy_name,
            pos.side.value,
            exit_price,
            reason.value,
            net_pnl,
            fees,
        )
        self._position = None

    # ---- 캔들 기반 SL/TP 체결 시뮬 (백테/페이퍼 공통) ----

    def check_candle_sl_tp(
        self,
        position: Position,
        candle_high: float,
        candle_low: float,
    ) -> tuple[float, ExitReason] | None:
        """한 캔들 내 SL/TP 도달 판정. 동시 도달 시 SL 우선 (정책 (a))."""
        if position is None:
            return None
        sl = position.stop_loss
        tp = position.take_profit
        if sl is None and tp is None:
            return None

        if position.side == PositionSide.LONG:
            sl_hit = sl is not None and candle_low <= sl
            tp_hit = tp is not None and candle_high >= tp
            if sl_hit:
                return sl, ExitReason.SL_HIT
            if tp_hit:
                return tp, ExitReason.TP_HIT
        elif position.side == PositionSide.SHORT:
            sl_hit = sl is not None and candle_high >= sl
            tp_hit = tp is not None and candle_low <= tp
            if sl_hit:
                return sl, ExitReason.SL_HIT
            if tp_hit:
                return tp, ExitReason.TP_HIT
        return None

    # ---- 전략 훅: update_stop_loss / should_force_exit ----

    def check_strategy_exits(
        self,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
    ) -> ExitDecision | None:
        """보유 중 봉 마감 시 호출.

        1) update_stop_loss 결과로 position.stop_loss 갱신 (None 반환=유지)
        2) should_force_exit가 ExitDecision 반환 시 그 값 리턴 (엔진이 청산 트리거)

        orphan(소속 전략이 active 아님)이면 두 훅 모두 스킵.
        """
        if self._position is None:
            return None
        strategy = self.strategy_by_name.get(self._position.strategy_name)
        if strategy is None:
            return None

        ctx = self._build_ctx(
            strategy, candles_per_tf, current_price, balance, now
        )
        try:
            new_sl = strategy.update_stop_loss(ctx, self._position)
            if new_sl is not None:
                self._position.stop_loss = float(new_sl)
        except Exception as e:
            logger.error(
                "update_stop_loss hook error in %s: %s",
                strategy.name,
                e,
                exc_info=True,
            )

        try:
            return strategy.should_force_exit(ctx, self._position)
        except Exception as e:
            logger.error(
                "should_force_exit hook error in %s: %s",
                strategy.name,
                e,
                exc_info=True,
            )
            return None
