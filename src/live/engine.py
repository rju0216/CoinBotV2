"""라이브·페이퍼 실시간 엔진 (CoreEngine).

AbstractEngine을 상속하여 DataFeed의 BAR_CLOSED 이벤트로 구동한다.
재시작 시 거래소 포지션과 DB의 open trades를 매칭하여 Position을 복원
(자동 입양 정책 7-1). 뼈대 상태(전략 0개)에서 거래소 포지션이 있으면
에러로 중단한다 (정책 7 (a)).

funding fee는 close 직전 fetch_funding_history로 조회하여 FeeModel의
PnL 정산에 주입한다.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.core.engine_base import AbstractEngine
from src.core.enums import (
    EventType,
    ExitReason,
    PositionSide,
    PositionStatus,
)
from src.core.types import Position
from src.data.feed import DataFeed
from src.data.store import DataStore
from src.live.oos_monitor import LiveOOSMonitor

logger = logging.getLogger(__name__)


def _candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


class CoreEngine(AbstractEngine):
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        if mode not in ("live", "paper"):
            raise ValueError(f"CoreEngine only supports live/paper, got: {mode}")
        super().__init__(config, mode=mode)
        # 라이브/페이퍼는 DataStore를 사용해 거래·잔액을 영속 기록.
        self.data_store = DataStore(config, mode)
        self.data_feed: DataFeed | None = None
        self._stop = asyncio.Event()
        # (I-005) ccxt.pro watch_ohlcv가 진행 중 봉을 재발행할 수 있어,
        # TF별 마지막 처리 타임스탬프를 유지해 중복 전략 평가를 차단.
        self._processed_bars: dict[str, int] = {}

    # ---- 초기화 / 종료 ----

    async def initialize(self) -> None:
        await self.data_store.initialize()
        await self.broker.initialize()
        await self._restore_state()
        self.data_feed = DataFeed(
            self.config, self.event_bus, timeframes=self.timeframes
        )
        await self._backfill_candles()
        # BP-2-3: OOS monitor 초기화 (config.live.oos_monitoring.enabled=true일 때만 활성)
        oos_cfg = (self.config.get("live", {}) or {}).get(
            "oos_monitoring", {}
        ) or {}
        if oos_cfg.get("enabled", False):
            self.oos_monitor = LiveOOSMonitor(self.config)
            logger.info(
                "OOS monitor enabled: window=%d, horizon=%d, threshold=%.3f",
                self.oos_monitor.window,
                self.oos_monitor.horizon,
                self.oos_monitor.min_acc_threshold,
            )

    async def shutdown(self) -> None:
        self._stop.set()
        if self.data_feed is not None:
            await self.data_feed.close()
        await self.broker.close()
        await self.data_store.close()

    # ---- 상태 복원 (잠재 이슈 I-001/I-002 해결) ----

    async def _restore_state(self) -> None:
        """재시작 시 잔액·포지션·DD락 복원.

        포지션 매칭 정책:
          - 거래소 O + DB O + strategy_name match:
              - active 리스트에 있으면 정상 OPEN, 없으면 ORPHAN
          - 거래소 O + DB ∅ + 전략 0개: 에러 중단 (정책 7 (a))
          - 거래소 O + DB ∅ + 전략 ≥1: strategy_name="_unknown" ORPHAN
          - 거래소 ∅ + DB O: DB의 open trades 사후 closed 처리
          - 거래소 ∅ + DB ∅: 정상 빈 슬롯
        """
        # 잔액/peak 복원
        balance = await self.broker.get_balance()
        initial = await self.data_store.get_initial_balance()
        if initial is None:
            await self.data_store.set_initial_balance(balance)
            initial = balance
        self.risk_manager.set_initial_balance(initial)
        peak = await self.data_store.get_peak_equity()
        if peak > 0:
            self.risk_manager.peak_equity = peak
        self.risk_manager.update_equity(balance)

        # 포지션 매칭
        exchange_pos = await self.broker.get_position()
        open_trades = await self.data_store.get_open_trades()

        # 1) 거래소 없음 + DB 없음
        if exchange_pos is None and not open_trades:
            logger.info("Clean startup: no open position")
            return

        # 2) 거래소 없음 + DB 있음 → DB의 open trades 사후 청산 처리
        if exchange_pos is None and open_trades:
            logger.warning(
                "DB has %d open trades but exchange has none. Closing them.",
                len(open_trades),
            )
            for trade in open_trades:
                await self.data_store.close_trade(
                    trade_id=trade["id"],
                    exit_price=trade["entry_price"],
                    pnl=0.0,
                    pnl_pct=0.0,
                    exit_reason=ExitReason.ENGINE_SHUTDOWN.value,
                )
            return

        # 3) 거래소 있음 + 전략 0개 → 에러 중단 (정책 7 (a))
        if exchange_pos is not None and not self.strategies:
            raise RuntimeError(
                "Exchange has an open position but no active strategies "
                "configured. Either add strategies to config.strategies.active "
                "or close the exchange position manually before starting. "
                f"Position: side={exchange_pos['side'].value}, "
                f"size={exchange_pos['size']}, entry={exchange_pos['entry_price']}"
            )

        # 4) 거래소 있음 + DB 매칭 시도
        matched = self._match_trade_to_exchange(open_trades, exchange_pos)

        strategy_name: str
        sl_price: float | None
        tp_price: float | None
        trade_id: int | None
        entry_time: datetime

        if matched is not None:
            strategy_name = matched["strategy_name"]
            sl_price = matched.get("stop_loss")
            tp_price = matched.get("take_profit")
            trade_id = matched["id"]
            try:
                entry_time = datetime.fromisoformat(matched["timestamp"])
            except Exception:
                entry_time = datetime.now(timezone.utc)
        else:
            # 거래소엔 있으나 DB 매칭 실패 → unknown orphan
            logger.warning(
                "Exchange position has no matching DB trade: %s", exchange_pos
            )
            strategy_name = "_unknown"
            sl_price = None
            tp_price = None
            trade_id = None
            entry_time = datetime.now(timezone.utc)

        # 자동 입양 (7-1): active 리스트에 있으면 OPEN, 없으면 ORPHAN
        status = (
            PositionStatus.OPEN
            if strategy_name in self.strategy_by_name
            else PositionStatus.ORPHAN
        )
        if status == PositionStatus.ORPHAN:
            logger.warning(
                "Adopted as orphan: strategy '%s' not in active list. "
                "Engine-level SL/TP will apply; strategy-specific hooks "
                "(should_force_exit, update_stop_loss) will be skipped.",
                strategy_name,
            )

        self._position = Position(
            side=exchange_pos["side"],
            size=exchange_pos["size"],
            entry_price=exchange_pos["entry_price"],
            entry_time=entry_time,
            strategy_name=strategy_name,
            stop_loss=sl_price,
            take_profit=tp_price,
            trade_id=trade_id,
            status=status,
        )
        logger.info(
            "Restored position: [%s] %s %.4f @ %.2f (status=%s, trade_id=%s)",
            strategy_name,
            self._position.side.value,
            self._position.size,
            self._position.entry_price,
            status.value,
            trade_id,
        )

    @staticmethod
    def _match_trade_to_exchange(
        open_trades: list[dict], exchange_pos: dict
    ) -> dict | None:
        """거래소 포지션과 DB open trade 매칭: side + size 기준."""
        ex_side: PositionSide = exchange_pos["side"]
        ex_size = float(exchange_pos["size"])
        for trade in open_trades:
            try:
                trade_side = PositionSide(trade["side"])
            except ValueError:
                continue
            if trade_side == ex_side and abs(float(trade["size"]) - ex_size) < 1e-6:
                return trade
        return None

    # ---- 백필 ----

    async def _backfill_candles(self) -> None:
        assert self.data_feed is not None
        result = await self.data_feed.backfill()
        for tf, candles in result.items():
            df = _candles_to_df(candles)
            self.data_store.set_dataframe(tf, df)
        logger.info("Backfilled candles for timeframes: %s", list(result.keys()))

    # ---- 메인 루프 ----

    async def run(self) -> None:
        if self.data_feed is None:
            raise RuntimeError("Engine not initialized; call initialize() first")

        self.event_bus.subscribe(
            EventType.BAR_CLOSED.value, self._on_bar_closed
        )
        feed_task = asyncio.create_task(self.data_feed.stream())
        logger.info(
            "CoreEngine [%s] running. Active strategies: %s",
            self.mode,
            [s.name for s in self.strategies],
        )

        # 종료 이벤트 또는 feed 종료까지 대기
        done, pending = await asyncio.wait(
            [feed_task, asyncio.create_task(self._stop.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        logger.info("CoreEngine run loop exited")

    # ---- 봉 마감 핸들러 ----

    def _should_process_bar(self, tf: str, ts_ms: int) -> bool:
        """같은 TF에서 마지막에 본 ts보다 크지 않으면 진행 중(또는 중복) 이벤트로 간주.
        새 ts로 갱신되어야만 전략 평가를 수행한다.
        """
        last = self._processed_bars.get(tf, -1)
        if ts_ms <= last:
            return False
        self._processed_bars[tf] = ts_ms
        return True

    async def _on_bar_closed(self, data: dict) -> None:
        tf = data["timeframe"]
        candle = data["candle"]
        ts_ms = int(candle["timestamp"])

        # 최신 가격 반영은 매 발행마다 수행 (DataFrame 갱신)
        try:
            self.data_store.append_candle(tf, candle)
        except Exception as e:
            logger.error("append_candle failed: %s", e, exc_info=True)
            return

        # 진행 중 봉 재발행이면 전략 평가 skip (I-005)
        if not self._should_process_bar(tf, ts_ms):
            return

        candles_slice = {t: self.data_store.get_df(t) for t in self.timeframes}
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        now = pd.to_datetime(
            candle["timestamp"], unit="ms", utc=True
        ).to_pydatetime()

        # 1) SL/TP 캔들 체결 검사 (엔진 담당 정책 (a))
        if self._position is not None:
            fill = self.check_candle_sl_tp(self._position, high, low)
            if fill is not None:
                exit_price, reason = fill
                await self._close_with_funding(exit_price, reason, now)

        # 2) 전략 강제 청산 훅 (보유 중 & orphan 아님일 때만)
        balance = await self.broker.get_balance()
        if self._position is not None:
            decision = self.check_strategy_exits(
                candles_slice, close, balance, now
            )
            if decision is not None:
                await self._close_with_funding(close, decision.reason, now)
                balance = await self.broker.get_balance()

        # 3) 봉 마감 dispatch (entry/pyramid/reverse 평가)
        await self.evaluate_strategies_on_bar(
            tf, candles_slice, close, balance, now
        )

        # BP-2-3: OOS monitor 평가 (horizon 도달한 pending prediction 채점)
        if self.oos_monitor is not None:
            try:
                self.oos_monitor.evaluate_pending(now, close)
            except Exception as e:
                logger.warning("oos_monitor.evaluate_pending failed: %s", e)

        # 4) equity 로깅
        try:
            balance = await self.broker.get_balance()
            await self.data_store.log_equity(balance)
        except Exception as e:
            logger.warning("log_equity failed: %s", e)

    # ---- funding fee 조회 + close 래퍼 ----

    async def _close_with_funding(
        self, exit_price: float, reason: ExitReason, now: datetime
    ) -> None:
        funding = 0.0
        if self.mode == "live" and self._position is not None:
            funding = await self._fetch_funding_since_entry()
        await self.close_position(
            exit_price, reason, funding_fee=funding, now=now
        )

    # ---- 거래 기록 (DataStore 기반) ----

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
        return await self.data_store.log_trade(
            strategy_name=strategy_name,
            side=side.value,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

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
        await self.data_store.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            trading_fee=trading_fee,
            funding_fee=funding_fee,
            exit_reason=exit_reason,
        )

    async def _fetch_funding_since_entry(self) -> float:
        if self._position is None or self._position.entry_time is None:
            return 0.0
        try:
            records = await self.broker.fetch_funding_history(
                since=self._position.entry_time.isoformat(),
            )
            return sum(abs(float(r.get("amount", 0))) for r in records)
        except Exception as e:
            logger.warning("fetch_funding_history failed: %s", e)
            return 0.0
