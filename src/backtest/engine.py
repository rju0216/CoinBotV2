"""백테스트 엔진 — 봉마감 평가·진입·청산·SL/TP 시뮬을 한 클래스에 담은 골격.

전략(플러그인)의 존재를 `StrategyModule` 인터페이스로만 안다. 모든 거래 정책
(진입 신호·사이징·SL/TP·reverse·진입 게이트)은 전략이 소유하고 엔진은 집행만 한다.
체결은 캔들 가격으로 시뮬하고 PnL 은 `FeeModel.calc_pnl` 단일 공식으로 정산한다
(외부 브로커·거래소 없음 — 순수 인메모리).

흐름 (마스터 TF = 가장 작은 활성 TF 캔들 순회):
  1) 보유 중이면 SL/TP 캔들 체결 검사 (동시 hit 은 모델의 sl_tp_fill_priority)
  2) 보유 중이면 update_stop_loss / should_force_exit 훅
  3) 봉 경계 TF 별 evaluate_strategies_on_bar (진입 또는 reverse)
종료 시 잔여 포지션은 ENGINE_SHUTDOWN 사유로 강제 청산.

**용량 = N-트랜치 (Phase 6 (라), config ``backtest.max_slots``, 기본 1)**
슬롯을 N개까지 열어 각 트랜치가 **독립 진입가·SL/TP·만기**를 갖는다. 트랜치는 서로
간섭하지 않고 손익은 가산적이다 — 선형 perp 에서 거래소 넷팅(평균단가 1포지션)과
**손익이 정확히 동일**하므로 실거래 재현 가능(트랜치 장부는 봇이 보유, 청산은 reduce-only).
``max_slots=1`` 이면 기존 단일슬롯 동작과 **완전 동일**(가역·비파괴, 등가성 회귀로 박제).
사이징(총노출 N등분 여부)은 **모델 소유** — 엔진은 용량만 안다.
reverse 는 ``max_slots=1`` 에서만 평가된다(N>1 + allow_reverse 는 구성 시 하드에러).

lookahead 차단: `_slice_candles(ts)` 가 ts '미만' 캔들만 전달 → 진행 중 봉의 close 가
피처에 새지 않는다. 결과적으로 `ctx.candles[tf].iloc[-1]` 은 항상 직전 마감 봉.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.accounting.account_tracker import AccountTracker
from src.accounting.fee_model import FeeModel
from src.core.enums import ExitReason, PositionSide, PositionStatus, SignalSide
from src.core.types import (
    AccountState,
    ExitDecision,
    Position,
    Signal,
    StrategyContext,
)
from src.data.historical import TF_MS, HistoricalDataLoader
from src.strategy.base import StrategyModule
from src.strategy.registry import load_active_strategies

logger = logging.getLogger(__name__)

REPORT_BASE = Path("data/backtest_reports")

_TF_PRIORITY = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}


def signal_side_to_position_side(side: SignalSide) -> PositionSide:
    if side == SignalSide.LONG:
        return PositionSide.LONG
    if side == SignalSide.SHORT:
        return PositionSide.SHORT
    return PositionSide.NONE


class BacktestEngine:
    def __init__(
        self,
        config: dict[str, Any],
        start: str | datetime,
        end: str | datetime,
    ) -> None:
        self.config = config
        self.fee_model = FeeModel.from_config(config)
        self.account_tracker = AccountTracker()
        self.strategies: list[StrategyModule] = load_active_strategies(config)
        self.strategy_by_name: dict[str, StrategyModule] = {
            s.name: s for s in self.strategies
        }
        self.timeframes: list[str] = self._compute_timeframe_union()
        self.master_timeframe: str | None = (
            self.timeframes[0] if self.timeframes else None
        )

        self.start_dt = self._parse_dt(start)
        self.end_dt = self._parse_dt(end)
        self.balance = float(
            (config.get("backtest", {}) or {}).get("initial_balance", 10000.0)
        )

        # 배타적 경합 — 전역 슬롯 max_slots 개 (기본 1 = 기존 단일슬롯 동작).
        bt_cfg = config.get("backtest", {}) or {}
        self.max_slots = int(bt_cfg.get("max_slots", 1))
        if self.max_slots < 1:
            raise ValueError(f"backtest.max_slots 는 1 이상이어야 함: {self.max_slots}")
        self._positions: list[Position] = []
        if self.max_slots > 1:
            # reverse 는 단일슬롯 전제 로직 → N>1 에서 조용히 무시되지 않도록 하드페일.
            reversing = [s.name for s in self.strategies
                         if getattr(s, "allow_reverse", False)]
            if reversing:
                raise ValueError(
                    f"max_slots>1 과 allow_reverse 는 함께 쓸 수 없음: {reversing} "
                    "(reverse 는 단일슬롯에서만 평가된다)"
                )

        self.candles_per_tf: dict[str, pd.DataFrame] = {}
        self.equity_curve: list[tuple[datetime, float]] = []
        # 미실현 포함 mark-to-market 곡선(가산적 부가 산출 — equity_curve 는 불변).
        # 트랜치가 여러 개 열린 구간의 낙폭은 실현잔고 곡선에 안 보이므로 MDD 계측용.
        self.equity_curve_mtm: list[tuple[datetime, float]] = []
        self._next_trade_id = 0
        self._open_trades: dict[int, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []

    @property
    def position(self) -> Position | None:
        """가장 오래된(첫) 트랜치. max_slots=1 이면 기존 의미와 동일."""
        return self._positions[0] if self._positions else None

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def has_capacity(self) -> bool:
        return len(self._positions) < self.max_slots

    @staticmethod
    def _parse_dt(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        s = str(value)
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = datetime.strptime(s, "%Y-%m-%d")
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    # ---- 활성 TF 산출 ----

    def _compute_timeframe_union(self) -> list[str]:
        """활성 전략의 entry_timeframe + required_timeframes 합집합.
        가장 작은 TF 가 첫 번째 (마스터 루프 후보)."""
        seen: set[str] = set()
        for s in self.strategies:
            if s.entry_timeframe:
                seen.add(s.entry_timeframe)
            for tf in s.required_timeframes:
                if tf:
                    seen.add(tf)
        return sorted(seen, key=lambda tf: _TF_PRIORITY.get(tf, 99))

    # ---- lifecycle ----

    def initialize(self) -> None:
        self.account_tracker.set_initial_balance(self.balance)
        self._load_candles()

    def run(self) -> None:
        if self.master_timeframe is None or not self.candles_per_tf:
            logger.error("Cannot run: no master timeframe or candles loaded")
            return
        master_df = self.candles_per_tf.get(self.master_timeframe)
        if master_df is None or master_df.empty:
            logger.warning("Master candles empty for %s", self.master_timeframe)
            return

        master_df = master_df.loc[
            (master_df.index >= pd.Timestamp(self.start_dt))
            & (master_df.index <= pd.Timestamp(self.end_dt))
        ]
        if master_df.empty:
            logger.warning(
                "No master candles in range %s ~ %s", self.start_dt, self.end_dt
            )
            return
        if not self.strategies:
            logger.warning(
                "Backtest with 0 active strategies — no trades will occur"
            )
        logger.info(
            "Backtest run: %d %s candles, strategies=%s",
            len(master_df),
            self.master_timeframe,
            [s.name for s in self.strategies],
        )

        for ts, candle in master_df.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            open_ = float(candle["open"])
            now = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

            # SL/TP — 봉 내 hit 여부는 high/low 로 판정 (거래소 체결 시뮬).
            # 트랜치별 독립 판정 (손익 가산적 → 순회 순서 무관).
            for pos in list(self._positions):
                fill = self.check_candle_sl_tp(pos, high, low)
                if fill is not None:
                    exit_price, reason = fill
                    self.close_position(pos, exit_price, reason, now=now)

            # ts 시점엔 직전 봉까지의 데이터로 평가 + open 가격으로 진입.
            # _slice_candles 는 ts 미만 슬라이스 (lookahead 제거).
            candles_slice = self._slice_candles(ts)

            for pos in list(self._positions):
                exit_decision = self.check_strategy_exits(
                    pos, candles_slice, open_, self.balance, now
                )
                if exit_decision is not None:
                    self.close_position(pos, open_, exit_decision.reason, now=now)

            for tf in self.timeframes:
                if self._is_tf_boundary(now, tf):
                    self.evaluate_strategies_on_bar(
                        tf, candles_slice, open_, self.balance, now
                    )

            self.equity_curve.append((now, self.balance))
            self.equity_curve_mtm.append(
                (now, self._mark_to_market(float(candle["close"])))
            )

        if self._positions:
            last_ts = master_df.index[-1]
            last_close = float(master_df["close"].iloc[-1])
            last_now = (
                last_ts.to_pydatetime()
                if hasattr(last_ts, "to_pydatetime")
                else last_ts
            )
            for pos in list(self._positions):
                self.close_position(
                    pos, last_close, ExitReason.ENGINE_SHUTDOWN, now=last_now
                )
            self.equity_curve.append((last_now, self.balance))
            self.equity_curve_mtm.append((last_now, self.balance))

        logger.info(
            "Backtest complete: balance=%.2f, trades=%d",
            self.balance,
            len(self.trades),
        )

    def shutdown(self) -> None:
        return None

    # ---- 컨텍스트 빌더 ----

    def _build_ctx(
        self,
        strategy: StrategyModule,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
        position: Position | None = None,
    ) -> StrategyContext:
        """position 지정 시 그 트랜치에 바인딩(훅 경로), 미지정 시 이 전략 소유 최신 트랜치.

        ``is_slot_occupied`` = **용량 소진 여부** (max_slots=1 이면 기존 의미와 동일)."""
        own_position = position
        if own_position is None:
            for pos in reversed(self._positions):
                if pos.strategy_name == strategy.name:
                    own_position = pos
                    break
        return StrategyContext(
            candles=dict(candles_per_tf),
            current_price=current_price,
            balance=balance,
            position=own_position,
            is_slot_occupied=not self.has_capacity,
            params=strategy.params,
            now=now,
            account=self._build_account_state(balance, current_price),
        )

    def _build_account_state(
        self, balance: float, current_price: float
    ) -> AccountState:
        """allow_entry / compute_position_size 가 참조할 계좌 텔레메트리 (계측치)."""
        equity = balance + self._unrealized(current_price)
        tracker = self.account_tracker
        return AccountState(
            balance=balance,
            equity=equity,
            peak_equity=tracker.peak_equity,
            daily_pnl=tracker.daily_pnl,
            initial_balance=tracker.initial_balance,
            drawdown_pct=tracker.drawdown_pct(equity),
        )

    def _unrealized(self, current_price: float) -> float:
        """열린 트랜치 전체의 미실현 손익 합 (수수료 미차감 — 계측치)."""
        if current_price <= 0:
            return 0.0
        total = 0.0
        for pos in self._positions:
            if pos.side == PositionSide.LONG:
                total += (current_price - pos.entry_price) * pos.size
            elif pos.side == PositionSide.SHORT:
                total += (pos.entry_price - current_price) * pos.size
        return total

    def _mark_to_market(self, price: float) -> float:
        """실현 잔고 + 미실현 = mark-to-market 자본 (equity_curve_mtm 용)."""
        return self.balance + self._unrealized(price)

    # ---- 봉 마감 dispatch ----

    def evaluate_strategies_on_bar(
        self,
        bar_close_tf: str,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
    ) -> None:
        """봉 마감 시 전략 평가.

        1) 관련 TF 전략에 on_bar_close 훅 dispatch
        2) 용량이 남으면 entry_timeframe 일치 전략을 우선순위로 generate_signal,
           첫 actionable 신호로 진입 시도 (봉당 최대 1 트랜치)
        3) 용량 소진 시 보유 전략의 should_reverse 로 청산·재진입 결정
           — **max_slots=1 에서만.** N>1 은 reverse 미평가(구성 시 하드페일로 방어).
        """
        # 자정 경계 인식 시 daily_pnl reset
        self.account_tracker.maybe_reset_for_new_day(now)

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

        # 2) 용량 남음 → 진입 시도. **전략당 최대 1 트랜치**, 용량이 남는 한 다음 전략도 시도.
        #    (max_slots=1 이면 첫 진입으로 용량이 소진돼 break — 기존 "첫 성공 시 return"과 동일.
        #     max_slots>1 이면 지평 앙상블처럼 여러 전략이 같은 봉에서 각자 1 트랜치를 연다.)
        if self.has_capacity:
            for strategy in self.strategies:
                if not self.has_capacity:
                    break
                if strategy.entry_timeframe != bar_close_tf:
                    continue
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                signal = strategy.generate_signal(ctx)
                if not signal.is_actionable:
                    continue
                self.try_enter(strategy, signal, ctx, now)
            return

        # 3) 용량 소진 → reverse 판정 (단일슬롯 전제 로직).
        if self.max_slots != 1:
            return
        held_position = self._positions[0]
        held_strategy = self.strategy_by_name.get(held_position.strategy_name)
        if held_strategy is not None:
            for strategy in self.strategies:
                if strategy.entry_timeframe != bar_close_tf:
                    continue
                ctx = self._build_ctx(
                    strategy, candles_per_tf, current_price, balance, now
                )
                signal = strategy.generate_signal(ctx)
                if not signal.is_actionable:
                    continue
                held_ctx = self._build_ctx(
                    held_strategy, candles_per_tf, current_price, balance, now
                )
                if held_strategy.should_reverse(held_ctx, held_position, signal):
                    self.close_position(
                        held_position, current_price, ExitReason.REVERSE_SIGNAL,
                        now=now,
                    )
                    # ★I-011★ 청산된 뒤에는 루프를 계속하면 안 된다 — held_position 은
                    # 이미 닫혔으므로 다음 전략에 대해 should_reverse 를 재평가하면
                    # 죽은 포지션 기준 판정 + 이중 진입이 된다. 재진입 성패와 무관하게 종료.
                    self.try_enter(strategy, signal, ctx, now)
                    return

    # ---- 진입 ----

    def try_enter(
        self,
        strategy: StrategyModule,
        signal: Signal,
        ctx: StrategyContext,
        now: datetime,
    ) -> bool:
        if not self.has_capacity:
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

        # 체결은 캔들 가격(current_price = 봉 open)으로 시뮬. 수수료는 청산 시 왕복 차감.
        position_side = signal_side_to_position_side(signal.side)
        entry_price = ctx.current_price

        trade_id = self._record_trade_open(
            strategy_name=strategy.name,
            side=position_side,
            size=size,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            now=now,
        )

        position = Position(
            side=position_side,
            size=size,
            entry_price=entry_price,
            entry_time=now,
            strategy_name=strategy.name,
            stop_loss=sl_price,
            take_profit=tp_price,
            trade_id=trade_id,
            status=PositionStatus.OPEN,
        )
        self._positions.append(position)

        try:
            strategy.on_position_opened(position)
        except Exception as e:
            logger.error(
                "on_position_opened hook error in %s: %s", strategy.name, e,
                exc_info=True,
            )

        logger.info(
            "ENTRY[%s]: %s %.4f @ %.2f, SL=%s, TP=%s, trade_id=%d",
            strategy.name, position_side.value, size, entry_price,
            f"{sl_price:.2f}" if sl_price is not None else "None",
            f"{tp_price:.2f}" if tp_price is not None else "None",
            trade_id,
        )
        return True

    # ---- 청산 ----

    def close_position(
        self,
        position: Position,
        exit_price: float,
        reason: ExitReason,
        funding_fee: float = 0.0,
        now: datetime | None = None,
    ) -> None:
        """지정 트랜치 하나만 청산 (나머지 트랜치는 그대로 유지).

        Position 은 dataclass(값 비교)이므로 **동일성(is)** 으로 대조·제거한다."""
        if position is None or not any(p is position for p in self._positions):
            return
        pos = position
        now = now or datetime.now(timezone.utc)

        # 수수료·PnL 정산 — FeeModel 단일 공식. 왕복 수수료를 balance 에서 차감.
        fees = self.fee_model.estimate_round_trip(
            pos.entry_price, exit_price, pos.size
        )
        pnl_result = self.fee_model.calc_pnl(
            pos.side, pos.entry_price, exit_price, pos.size,
            fees=fees, funding=funding_fee,
        )
        net_pnl = pnl_result["net_pnl"]
        self.balance += net_pnl

        if pos.trade_id is not None:
            self._record_trade_close(
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

        # 계좌 추적 갱신 (계측)
        self.account_tracker.add_pnl(net_pnl)
        self.account_tracker.update_equity(self.balance)

        strategy = self.strategy_by_name.get(pos.strategy_name)
        if strategy is not None:
            try:
                strategy.on_position_closed(pos, net_pnl)
            except Exception as e:
                logger.error(
                    "on_position_closed hook error in %s: %s", strategy.name, e,
                    exc_info=True,
                )

        logger.info(
            "EXIT[%s]: %s @ %.2f, reason=%s, net_pnl=$%.2f, fees=$%.2f",
            pos.strategy_name, pos.side.value, exit_price, reason.value,
            net_pnl, fees,
        )
        self._positions = [p for p in self._positions if p is not pos]

    # ---- 캔들 기반 SL/TP 체결 시뮬 ----

    def check_candle_sl_tp(
        self, position: Position, candle_high: float, candle_low: float
    ) -> tuple[float, ExitReason] | None:
        """한 캔들 내 SL/TP 도달 판정 (거래소 체결 시뮬).

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
        """동시 도달 시 체결 우선순위 (보유 전략 소유). 부재 시 보수적 sl_first."""
        strat = self.strategy_by_name.get(position.strategy_name)
        return getattr(strat, "sl_tp_fill_priority", "sl_first") if strat else "sl_first"

    # ---- 전략 훅: update_stop_loss / should_force_exit ----

    def check_strategy_exits(
        self,
        position: Position,
        candles_per_tf: dict[str, pd.DataFrame],
        current_price: float,
        balance: float,
        now: datetime,
    ) -> ExitDecision | None:
        """보유 중 봉 마감 시 **트랜치별로** 호출.

        1) update_stop_loss 결과로 position.stop_loss 갱신 (None 반환=유지)
        2) should_force_exit 가 ExitDecision 반환 시 그 값 리턴 (엔진이 청산 트리거)
        """
        if position is None:
            return None
        strategy = self.strategy_by_name.get(position.strategy_name)
        if strategy is None:
            return None

        ctx = self._build_ctx(
            strategy, candles_per_tf, current_price, balance, now, position=position
        )
        try:
            new_sl = strategy.update_stop_loss(ctx, position)
            if new_sl is not None:
                position.stop_loss = float(new_sl)
        except Exception as e:
            logger.error(
                "update_stop_loss hook error in %s: %s", strategy.name, e,
                exc_info=True,
            )

        try:
            return strategy.should_force_exit(ctx, position)
        except Exception as e:
            logger.error(
                "should_force_exit hook error in %s: %s", strategy.name, e,
                exc_info=True,
            )
            return None

    # ---- 거래 기록 (인메모리) ----

    def _record_trade_open(
        self,
        strategy_name: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        now: datetime,
    ) -> int:
        self._next_trade_id += 1
        tid = self._next_trade_id
        self._open_trades[tid] = {
            "id": tid,
            "strategy_name": strategy_name,
            "side": side.value,
            "size": size,
            "entry_price": entry_price,
            "entry_time": now,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
        }
        return tid

    def _record_trade_close(
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
        rec = self._open_trades.pop(trade_id, None)
        if rec is None:
            rec = {
                "id": trade_id,
                "strategy_name": position.strategy_name,
                "side": position.side.value,
                "size": position.size,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
            }
        rec.update(
            {
                "exit_time": now,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "trading_fee": trading_fee,
                "funding_fee": funding_fee,
                "exit_reason": exit_reason,
                "status": "closed",
            }
        )
        self.trades.append(rec)

    # ---- 캔들 로딩 ----

    def _load_candles(self) -> None:
        # data.history_bars 만큼 warmup 캔들을 미리 로드해 indicator NaN 구간을 단축.
        warmup_bars = int((self.config.get("data", {}) or {}).get("history_bars", 300))
        loader = HistoricalDataLoader(self.config)
        try:
            end_ms = int(self.end_dt.timestamp() * 1000)
            for tf in self.timeframes:
                tf_ms = TF_MS.get(tf, 60_000)
                start_ms = int(self.start_dt.timestamp() * 1000) - warmup_bars * tf_ms
                df = loader.download_range_merged(tf, start_ms, end_ms)
                self.candles_per_tf[tf] = df
                logger.info(
                    "Loaded %d %s candles (warmup_bars=%d)", len(df), tf, warmup_bars
                )
        finally:
            loader.close()

    def _is_tf_boundary(self, ts: datetime, tf: str) -> bool:
        if tf == "1m":
            return ts.second == 0
        if tf == "5m":
            return ts.minute % 5 == 0 and ts.second == 0
        if tf == "15m":
            return ts.minute % 15 == 0 and ts.second == 0
        if tf == "1h":
            return ts.minute == 0 and ts.second == 0
        if tf == "4h":
            return ts.hour % 4 == 0 and ts.minute == 0 and ts.second == 0
        if tf == "1d":
            return ts.hour == 0 and ts.minute == 0 and ts.second == 0
        return False

    def _slice_candles(self, ts) -> dict[str, pd.DataFrame]:
        """ts 시점 직전까지의 캔들 반환 (lookahead 제거).

        진행 중 봉(index == ts)과 그 이후를 배제 — ts 시점엔 그 봉이 아직 마감 전이라
        close 가 확정되지 않았으므로, 그 값이 피처로 새면 미래 정보 누출이 된다.

        구현: 인덱스는 시간순 정렬(온-그리드 감사 데이터·시간순 주입 계약)이므로
        ``searchsorted(ts, "left")`` = ts '미만' 봉 수. ``iloc[:pos]`` 슬라이스(뷰)는
        ``df[df.index < ts]`` 부울마스크와 **결과 동일**하나 봉당 O(log n)이라 전체
        백테가 O(n²)→O(n log n) (긴/촘촘한 1m 백테 실행시간 급감, D-031). 등가성은
        test_lookahead 의 회귀로 박제.
        """
        result: dict[str, pd.DataFrame] = {}
        for tf, df in self.candles_per_tf.items():
            if df.empty:
                result[tf] = df
            else:
                pos = df.index.searchsorted(ts, side="left")
                result[tf] = df.iloc[:pos]
        return result

    # ---- 결과 ----

    def summary(self) -> dict[str, Any]:
        """핵심 성과 지표 dict (콘솔 출력 + metrics.json 공용)."""
        initial = self.account_tracker.initial_balance
        final = self.equity_curve[-1][1] if self.equity_curve else initial
        total_pnl = final - initial
        total_pct = (total_pnl / initial * 100.0) if initial > 0 else 0.0

        n_total = len(self.trades)
        winning = [t for t in self.trades if (t.get("pnl") or 0) > 0]
        losing = [t for t in self.trades if (t.get("pnl") or 0) < 0]
        gp = sum(t["pnl"] for t in winning) if winning else 0.0
        gl = abs(sum(t["pnl"] for t in losing)) if losing else 0.0
        pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        win_rate = (len(winning) / n_total * 100.0) if n_total > 0 else 0.0

        max_dd = 0.0
        if self.equity_curve:
            peak = self.equity_curve[0][1]
            for _, eq in self.equity_curve:
                peak = max(peak, eq)
                if peak > 0:
                    max_dd = max(max_dd, (peak - eq) / peak * 100.0)

        return {
            "initial_balance": round(initial, 2),
            "final_balance": round(final, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pct, 2),
            "num_trades": n_total,
            "num_winners": len(winning),
            "num_losers": len(losing),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
            "max_drawdown_pct": round(max_dd, 2),
        }

    def write_reports(self, out_dir: str | Path | None = None) -> Path:
        """결과 3종(trades.csv / equity_curve.csv / metrics.json)을 out_dir 에 저장.

        out_dir 미지정 시 `data/backtest_reports/backtest_{start}_{end}/` 사용.
        """
        if out_dir is None:
            out_dir = REPORT_BASE / (
                f"backtest_{self.start_dt.date()}_{self.end_dt.date()}"
            )
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        cols_order = [
            "id", "strategy_name", "side", "size",
            "entry_time", "entry_price", "exit_time", "exit_price",
            "stop_loss", "take_profit",
            "pnl", "pnl_pct", "trading_fee", "funding_fee",
            "exit_reason", "status",
        ]
        if self.trades:
            df = pd.DataFrame(self.trades)
            ordered = [c for c in cols_order if c in df.columns] + [
                c for c in df.columns if c not in cols_order
            ]
            df[ordered].to_csv(out / "trades.csv", index=False)
        else:
            (out / "trades.csv").write_text(
                ",".join(cols_order) + "\n", encoding="utf-8"
            )

        if self.equity_curve:
            ec = pd.DataFrame(self.equity_curve, columns=["timestamp", "balance"])
            ec.set_index("timestamp", inplace=True)
            ec.to_csv(out / "equity_curve.csv")
        else:
            (out / "equity_curve.csv").write_text(
                "timestamp,balance\n", encoding="utf-8"
            )

        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)

        return out

    # ---- 테스트용: 외부 캔들 주입 ----

    def inject_candles(self, candles_per_tf: dict[str, pd.DataFrame]) -> None:
        self.candles_per_tf = {tf: df.copy() for tf, df in candles_per_tf.items()}
