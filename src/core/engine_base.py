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

from src.accounting.account_tracker import AccountTracker
from src.accounting.fee_model import FeeModel
from src.core.enums import (
    EventType,
    ExitReason,
    PositionSide,
    PositionStatus,
    SignalSide,
)
from src.core.event_bus import EventBus
from src.core.types import (
    AccountState,
    ExitDecision,
    Position,
    Signal,
    StrategyContext,
)
from src.execution.broker import Broker
from src.strategy.base import StrategyModule
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
    # bar_context 생성 시 *직전 마감 봉* 의 df 인덱스.
    # 백테 default (-1): df = _slice_candles(ts) (ts 미만 슬라이스) → iloc[-1] = 직전 마감 봉.
    # CoreEngine override (-2): df = data_store.get_df() (ccxt watch_ohlcv 새 봉
    # single tick 포함) → iloc[-2] = 직전 마감 봉. 각 엔진이 자기 영역의 df 구조
    # 의미에 맞춰 1줄 override.
    LAST_CLOSED_BAR_IDX: int = -1

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        self.config = config
        self.mode = mode
        self.event_bus = EventBus()
        self.fee_model = FeeModel.from_config(config)
        self.account_tracker = AccountTracker()
        self.broker = Broker(config, mode)

        self.strategies: list[StrategyModule] = load_active_strategies(config)
        self.strategy_by_name: dict[str, StrategyModule] = {
            s.name: s for s in self.strategies
        }

        self.timeframes: list[str] = self._compute_timeframe_union()
        self.master_timeframe: str | None = (
            self.timeframes[0] if self.timeframes else None
        )

        # 배타적 경합 — 전역 슬롯 1개 (동시에 한 포지션만 보유)
        self._position: Position | None = None

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
        entry_order_id: str | None = None,
    ) -> int:
        """진입 기록. 반환값은 trade_id. entry_order_id 는 옵션."""

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
        exit_order_id: str | None = None,
    ) -> None:
        """청산 기록. exit_order_id 는 옵션."""

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
        # 라이브는 ctx.candles 의 iloc[-1] 이 진행 중 봉(ccxt watch_ohlcv 가
        # 봉 진행 중 발행하는 tick). LAST_CLOSED_BAR_IDX 기준으로 진행 중 봉을 잘라
        # plugin 이 항상 '직전 마감 봉까지' 만 보게 한다 → candles 를 직접 쓰는
        # plugin 의 라이브-백테 신호 일치. 백테(-1)는 n_drop=0 무변경,
        # 라이브(-2)는 마지막 1봉 제외. current_price 는 진입가용으로 현재가 유지.
        n_drop = -1 - self.LAST_CLOSED_BAR_IDX
        if n_drop > 0:
            candles_per_tf = {
                tf: (df.iloc[:-n_drop] if len(df) > n_drop else df.iloc[:0])
                for tf, df in candles_per_tf.items()
            }
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
            account=self._build_account_state(balance, current_price),
        )

    def _build_account_state(
        self, balance: float, current_price: float
    ) -> AccountState:
        """allow_entry/compute_position_size 가 참조할 계좌 텔레메트리 (계측치)."""
        unrealized = 0.0
        pos = self._position
        if pos is not None and current_price > 0:
            if pos.side == PositionSide.LONG:
                unrealized = (current_price - pos.entry_price) * pos.size
            elif pos.side == PositionSide.SHORT:
                unrealized = (pos.entry_price - current_price) * pos.size
        equity = balance + unrealized
        tracker = self.account_tracker
        return AccountState(
            balance=balance,
            equity=equity,
            peak_equity=tracker.peak_equity,
            daily_pnl=tracker.daily_pnl,
            initial_balance=tracker.initial_balance,
            drawdown_pct=tracker.drawdown_pct(equity),
        )

    # ---- bar_context 빌더 ----

    def _build_bar_context(
        self, df: pd.DataFrame | None, current_price: float,
    ) -> dict | None:
        """직전 마감 봉의 OHLC + 그 직전 봉 close 추출 → SIGNAL 로그용.

        `LAST_CLOSED_BAR_IDX` attribute 가 *직전 마감 봉* 인덱스 결정 (백테 -1, 라이브 -2).
        라이브 영역 (CoreEngine, -2) 은 ccxt watch_ohlcv 가 새 봉 시작 시점에 발행하는
        single tick (open=high=low=close) 이 iloc[-1] 에 들어가므로 iloc[-2] 가 직전 마감 봉.

        `current_price` 인자는 default 영역 미사용. 향후 표기 방식 확장 (예: 진입 가격
        같이 표기) 시 사용 reserved — 호출 영역 시그니처 일관 유지.

        Returns:
            {"close", "prev_close", "high", "low"} 또는 None (데이터 부족 시).
        """
        idx = self.LAST_CLOSED_BAR_IDX
        if df is None or len(df) < abs(idx):
            return None
        last_closed = df.iloc[idx]
        prev_idx = idx - 1
        prev_close: float | None = (
            float(df.iloc[prev_idx]["close"])
            if len(df) >= abs(prev_idx) else None
        )
        return {
            "close": float(last_closed["close"]),
            "prev_close": prev_close,
            "high": float(last_closed["high"]),
            "low": float(last_closed["low"]),
        }

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
        3) 슬롯이 차있으면 보유 전략의 should_reverse 로 reverse 여부 결정
        4) 슬롯이 차있고 보유 전략이 supports_pyramiding 이면 generate_pyramid_signal
        """
        # 자정 경계 인식 시 daily_pnl reset (백테/페이퍼/라이브 일관).
        if self.account_tracker.maybe_reset_for_new_day(now):
            logger.info(
                "[AccountTracker] new UTC day boundary — daily_pnl reset "
                "(last_reset_date=%s)",
                self.account_tracker.last_reset_date,
            )

        # bar_context — _log_signal_status 의 가격 컨텍스트 출력용.
        # 백테에선 _log_signal_status default no-op 이라 미사용 (오버헤드 거의 0).
        # 직전 마감 봉 영역 사용 (라이브 새 봉 single tick 영역 회피).
        bar_context = self._build_bar_context(
            candles_per_tf.get(bar_close_tf), current_price,
        )

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
                self._log_signal_status(strategy, signal, bar_context)
                if not signal.is_actionable:
                    continue
                if await self.try_enter(strategy, signal, ctx, now):
                    return  # 첫 진입 성공 시 종료
            return

        # 3) 슬롯 참 → 보유(owner) 전략이 reverse 여부를 결정 (정책=모델 소유).
        #    후보 전략이 actionable 신호를 내면, 보유 전략의 should_reverse 가
        #    True 일 때만 청산 후 후보 신호로 재진입. orphan(보유 전략 부재)이면 skip.
        held_strategy = self.strategy_by_name.get(self._position.strategy_name)
        if held_strategy is not None:
            for strategy in self.strategies:
                if strategy.entry_timeframe != bar_close_tf:
                    continue
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                signal = strategy.generate_signal(ctx)
                self._log_signal_status(strategy, signal, bar_context)
                if not signal.is_actionable:
                    continue
                held_ctx = self._build_ctx(
                    held_strategy, candles_per_tf, current_price, balance, now
                )
                if held_strategy.should_reverse(held_ctx, self._position, signal):
                    await self.close_position(
                        current_price, ExitReason.REVERSE_SIGNAL, now=now
                    )
                    if await self.try_enter(strategy, signal, ctx, now):
                        return

        # reverse 가 청산했으나 재진입에 실패한 경우 슬롯이 비어 있을 수 있음 → 종료
        if self._position is None:
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
                    # 피라미딩 hook은 받되 실 청산/추가 진입 처리는 미구현.
                    # 향후 add_to_position 흐름으로 확장 가능.

    # ---- 신호/포지션 모니터링 hook ----

    def _log_signal_status(
        self,
        strategy: StrategyModule,
        signal: Signal,
        bar_context: dict | None = None,
    ) -> None:
        """generate_signal 결과를 모니터링 로그로 출력하는 hook (default no-op).

        backtest는 매 봉 수만 줄 출력 회피 위해 default no-op. CoreEngine이
        override해서 라이브/페이퍼 모드에서만 INFO 출력.

        bar_context = {"close", "prev_close", "high", "low"} 또는 None.
        라이브 [SIGNAL] 로그에 가격 컨텍스트 출력에 사용.
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

        # Circuit breaker OPEN 시 새 진입 차단 (거래소 연결 안전장치)
        if getattr(self, "_circuit_breaker_open", False):
            return False

        # 진입 게이트(리스크 정책) — 모델 소유
        if not strategy.allow_entry(ctx):
            logger.info("allow_entry rejected entry for %s", strategy.name)
            return False

        # SL/TP — 모델 소유. SL None = standing SL 미설정 (청산은 should_force_exit).
        sl_raw = strategy.compute_stop_loss(ctx, signal)
        sl_price = float(sl_raw) if sl_raw is not None else None
        tp_raw = strategy.compute_take_profit(ctx, signal, sl_price)
        tp_price = float(tp_raw) if tp_raw is not None else None

        # 사이징(공식·레버리지) — 모델 소유
        size = float(strategy.compute_position_size(ctx, signal, sl_price))
        if size <= 0:
            logger.info("compute_position_size returned 0 for %s", strategy.name)
            return False

        # 진입 주문
        position_side = signal_side_to_position_side(signal.side)
        # paper 모드에서 호가창 가용 시 VWAP 침투 가격 사용 (silent fallback)
        order = await self.broker.open_position(
            position_side, size, fill_price=ctx.current_price,
            orderbook=getattr(self, "_latest_orderbook", None),
        )
        if not order:
            logger.error("Open order failed for %s", strategy.name)
            return False

        # 진입 order id 추출 (라이브 OKX exchange order id, paper fake id)
        entry_order_id = order.get("id") if isinstance(order, dict) else None

        # 거래 기록 (구체 엔진이 DB 또는 메모리에 저장)
        trade_id = await self._record_trade_open(
            strategy_name=strategy.name,
            side=position_side,
            size=size,
            entry_price=ctx.current_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            now=now,
            entry_order_id=entry_order_id,
        )

        # 거래소 SL/TP pending (페이퍼는 no-op). SL/TP None 이면 등록 skip.
        if sl_price is not None:
            await self.broker.place_stop_loss(position_side, sl_price, size)
        if tp_price is not None:
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
            entry_order_id=entry_order_id,
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
            "ENTRY[%s]: %s %.4f @ %.2f, SL=%s, TP=%s, trade_id=%d",
            strategy.name,
            position_side.value,
            size,
            ctx.current_price,
            f"{sl_price:.2f}" if sl_price is not None else "None",
            f"{tp_price:.2f}" if tp_price is not None else "None",
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
        # paper 모드에서 호가창 가용 시 VWAP 침투 가격 사용 (silent fallback)
        # broker.close_position이 exception 발생해도 시스템 상태 정리 진행.
        # 거래소 SL/TP 자동 청산 후 redundant close 시도 같은 case에서 ExchangeError가 발생해도
        # self._position 정리 + DB close + event publish가 안 되는 mismatch 방지.
        # 거래소 상태 검증 후: 거래소 ∅ → 이미 청산된 상태로 정상 진행 / 거래소 O → propagate.
        # close order id 추출 — paper/normal close 시 order dict 의 'id'. SL/TP 자동
        # 청산은 거래소 ∅ 검증 path 라 order id 없음 (sync 시 fetch_my_trades 로 매칭)
        exit_order_id: str | None = None
        try:
            close_order = await self.broker.close_position(
                pos.side, pos.size, fill_price=exit_price,
                orderbook=getattr(self, "_latest_orderbook", None),
            )
            if isinstance(close_order, dict):
                # skip path 는 {'info': {'already_closed': True}, ...} 반환 → id 없음
                if not close_order.get("info", {}).get("already_closed"):
                    exit_order_id = close_order.get("id")
        except Exception as e:
            logger.warning("broker.close_position failed: %s — verifying exchange state", e)
            try:
                actual = await self.broker.get_position()
            except Exception as inner:
                logger.error(
                    "Cannot verify exchange position state after close failure: %s. "
                    "Propagating original error — manual intervention may be required.",
                    inner,
                )
                raise e from inner
            if actual is not None:
                # 거래소에 포지션 살아있음 → 진짜 청산 실패. exception propagate
                logger.error(
                    "broker.close_position failed AND exchange still has position. "
                    "Manual intervention required.",
                )
                raise
            # 거래소 ∅ → 이미 청산된 상태. 시스템 상태 정리는 정상 진행
            logger.warning(
                "Exchange shows no position — treating as already closed (likely SL/TP "
                "auto-triggered). Proceeding with system state cleanup."
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
                exit_order_id=exit_order_id,
            )

        # 계좌 추적 갱신 (계측)
        self.account_tracker.add_pnl(net_pnl)
        try:
            balance = await self.broker.get_balance()
            self.account_tracker.update_equity(balance)
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
            # 텔레그램 EXIT 알림 보강용 키 (exit_price/pnl_pct/closed_at).
            # 키 추가만이라 기존 구독자 영향 0, 백테/페이퍼 무영향.
            {
                "position": pos,
                "pnl": net_pnl,
                "reason": reason.value,
                "exit_price": exit_price,
                "pnl_pct": pnl_result["pnl_pct"],
                "closed_at": now,
            },
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
        """한 캔들 내 SL/TP 도달 판정 (백테/페이퍼 체결 시뮬 — 거래소 흉내).

        동시 도달 시 보유 전략의 sl_tp_fill_priority 로 결정 (정책=모델 소유).
        SL/TP 값은 모델이 설정한 것이며, 이 메서드는 가격 교차 *감지*만 한다.
        """
        if position is None:
            return None
        sl = position.stop_loss
        tp = position.take_profit
        if sl is None and tp is None:
            return None

        if position.side == PositionSide.LONG:
            sl_hit = sl is not None and candle_low <= sl
            tp_hit = tp is not None and candle_high >= tp
        elif position.side == PositionSide.SHORT:
            sl_hit = sl is not None and candle_high >= sl
            tp_hit = tp is not None and candle_low <= tp
        else:
            return None

        if sl_hit and tp_hit:
            if self._fill_priority(position) == "tp_first":
                return tp, ExitReason.TP_HIT
            return sl, ExitReason.SL_HIT
        if sl_hit:
            return sl, ExitReason.SL_HIT
        if tp_hit:
            return tp, ExitReason.TP_HIT
        return None

    def _fill_priority(self, position: Position) -> str:
        """동시 도달 시 체결 우선순위 (보유 전략 소유). orphan 이면 보수적 sl_first."""
        strat = self.strategy_by_name.get(position.strategy_name)
        return getattr(strat, "sl_tp_fill_priority", "sl_first") if strat else "sl_first"

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
