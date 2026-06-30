"""라이브·페이퍼 실시간 엔진 (CoreEngine).

AbstractEngine을 상속하여 DataFeed의 BAR_CLOSED 이벤트로 구동한다.
재시작 시 거래소 포지션과 DB의 open trades를 매칭하여 Position을 복원한다
(자동 입양). 뼈대 상태(전략 0개)에서 거래소 포지션이 있으면 에러로 중단한다.

funding fee는 close 직전 fetch_funding_history로 조회하여 FeeModel의
PnL 정산에 주입한다.
"""

from __future__ import annotations

import asyncio
import json
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
from src.data.orderbook import OrderBookCollector
from src.utils.notifier import Notifier, build_notifier_from_config

logger = logging.getLogger(__name__)


def _fmt_dollar(value: float) -> str:
    """부호 + $ + 절대값 형식. 양수: +$X.XX, 음수: -$X.XX. 0: +$0.00.

    ACCOUNT 로그의 unrealized_pnl / total_balance_diff / daily_pnl 영역에서 사용.
    dd 영역은 항상 손실 표기 (텍스트 -$%.2f) 라 미사용.
    """
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def _fmt_hold(seconds: float) -> str:
    """보유 시간 포맷 (XhYYm). [POSITION] 로그와 EXIT 알림이 공유한다.

    음수 방어 — abs + 부호 prefix 로 정확 표기 (-1h52m). 정상 매칭 시
    음수는 발생 안 하나, orphan(entry_time=now) 등 이상 case 에서 floor division
    오표기 방지. 음수 hold 자체가 이상 신호이므로 0 clamp 대신 정확 표기.
    """
    neg = seconds < 0
    s = abs(int(seconds))
    return f"{'-' if neg else ''}{s // 3600}h{(s % 3600) // 60:02d}m"


def _build_entry_message(pos) -> str:
    """ENTRY 텔레그램 본문 — 콘솔 [POSITION] 수준 정보량.

    SL/TP Δ% 는 entry_price 대비 (진입 시점 기준; 콘솔 [POSITION]은 current 대비).
    SL/TP 가 None (orphan 등) 이면 해당 부분 생략. plain text.

    샘플:
      LONG 0.0149 @ 67100.00
      SL=66500.00 (-0.89%) TP=68000.00 (+1.34%)
    """
    msg = f"{pos.side.value} {pos.size:.4f} @ {pos.entry_price:.2f}"
    parts = []
    if pos.stop_loss is not None and pos.entry_price > 0:
        sl_delta = (pos.stop_loss - pos.entry_price) / pos.entry_price * 100
        parts.append(f"SL={pos.stop_loss:.2f} ({sl_delta:+.2f}%)")
    if pos.take_profit is not None and pos.entry_price > 0:
        tp_delta = (pos.take_profit - pos.entry_price) / pos.entry_price * 100
        parts.append(f"TP={pos.take_profit:.2f} ({tp_delta:+.2f}%)")
    if parts:
        msg += "\n" + " ".join(parts)
    return msg


def _build_exit_message(pos, pnl, exit_price=None, pnl_pct=None, closed_at=None) -> str:
    """EXIT 텔레그램 본문 — entry→exit 가격 / net_pnl / pnl% / 보유 시간.

    exit_price / pnl_pct / closed_at 는 POSITION_CLOSED payload 키.
    None 이면 해당 부분 생략 (구버전 payload·orphan 안전). plain text.

    샘플:
      LONG 0.0149 @ 67100.00 → 68000.00
      net_pnl=+$13.41 (+1.34%) | 1h32m held
    """
    line1 = f"{pos.side.value} {pos.size:.4f} @ {pos.entry_price:.2f}"
    if exit_price is not None:
        line1 += f" → {exit_price:.2f}"
    pnl_str = _fmt_dollar(pnl)
    if pnl_pct is not None:
        pnl_str += f" ({pnl_pct:+.2f}%)"
    line2 = f"net_pnl={pnl_str}"
    if closed_at is not None and pos.entry_time is not None:
        held = (closed_at - pos.entry_time).total_seconds()
        line2 += f" | {_fmt_hold(held)} held"
    return f"{line1}\n{line2}"


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
    # 라이브 영역의 df 구조 — ccxt watch_ohlcv 가 새 봉 시작 시점에 single tick
    # (open=high=low=close) 으로 발행 → append_candle 이 iloc[-1] 에 single tick row 추가.
    # 따라서 *직전 마감 봉* 은 iloc[-2]. bar_context 의 close/high/low 가 진정한 봉 OHLC 영역.
    LAST_CLOSED_BAR_IDX = -2

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        if mode not in ("live", "paper"):
            raise ValueError(f"CoreEngine only supports live/paper, got: {mode}")
        super().__init__(config, mode=mode)
        # 라이브/페이퍼는 DataStore를 사용해 거래·잔액을 영속 기록.
        self.data_store = DataStore(config, mode)
        self.data_feed: DataFeed | None = None
        self._stop = asyncio.Event()
        # ccxt.pro watch_ohlcv가 진행 중 봉을 재발행할 수 있어,
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

        # notifier 인프라 초기화 + EventBus subscribe
        self.notifier: Notifier = build_notifier_from_config(self.config)
        self._setup_notifier_subscriptions()
        # Circuit breaker 발동 시 새 진입 차단
        self._circuit_breaker_open: bool = False

        # 호가창 collector 초기화 (config.live.orderbook.enabled=true 시)
        # paper 모드에서만 의미. live 모드는 거래소가 자동 처리.
        # collector는 ccxt async 클라이언트 필요 — DataFeed의 exchange 재사용
        self.orderbook_collector: OrderBookCollector | None = None
        ob_cfg = (self.config.get("live", {}) or {}).get("orderbook", {}) or {}
        if ob_cfg.get("enabled", False) and self.data_feed is not None:
            self.orderbook_collector = OrderBookCollector(
                self.config, self.data_feed.exchange,
            )
            logger.info(
                "OrderBook collector enabled: depth=%d, save_dir=%s",
                self.orderbook_collector.depth, self.orderbook_collector.save_dir,
            )
        # 최신 호가창 캐시 (BAR_CLOSED 시 fetch 후 try_enter/close_position에 전달)
        self._latest_orderbook: dict | None = None

    def _setup_notifier_subscriptions(self) -> None:
        """주요 EventType → notifier 송신 라우팅.

        levels config로 각 이벤트의 송신 활성/비활성 결정.
        """
        notif_cfg = (self.config.get("live", {}) or {}).get(
            "notifications", {}
        ) or {}
        levels = notif_cfg.get("levels", {}) or {}

        async def _on_circuit_breaker(data):
            if not levels.get("circuit_breaker", True):
                return
            self._circuit_breaker_open = True
            await self.notifier.send(
                "ERROR",
                "Circuit breaker OPEN",
                f"Consecutive API failures reached threshold. "
                f"New entries blocked. Manual reset required.",
                **data,
            )

        async def _on_position_opened(pos):
            if not levels.get("position_open", True):  # default true
                return
            # SL/TP 가격 + entry 대비 Δ% 보강 (_build_entry_message)
            await self.notifier.send(
                "INFO",
                f"ENTRY [{pos.strategy_name}]",
                _build_entry_message(pos),
                strategy=pos.strategy_name,
                side=pos.side.value,
                size=pos.size,
                entry_price=pos.entry_price,
            )

        async def _on_position_closed(data):
            if not levels.get("position_close", True):  # default true
                return
            pos = data.get("position")
            pnl = data.get("pnl", 0)
            reason = data.get("reason", "")
            if pos is None:
                return
            # entry→exit 가격 / pnl% / 보유 시간 보강 (_build_exit_message)
            await self.notifier.send(
                "INFO",
                f"EXIT [{pos.strategy_name}] {reason}",
                _build_exit_message(
                    pos, pnl,
                    exit_price=data.get("exit_price"),
                    pnl_pct=data.get("pnl_pct"),
                    closed_at=data.get("closed_at"),
                ),
                strategy=pos.strategy_name,
                pnl=pnl,
                reason=reason,
            )

        self.event_bus.subscribe("circuit_breaker_open", _on_circuit_breaker)
        self.event_bus.subscribe(EventType.POSITION_OPENED.value, _on_position_opened)
        self.event_bus.subscribe(EventType.POSITION_CLOSED.value, _on_position_closed)

    async def shutdown(self) -> None:
        self._stop.set()
        if self.data_feed is not None:
            await self.data_feed.close()
        await self.broker.close()
        await self.data_store.close()

    # ---- 상태 복원 ----

    async def _restore_daily_pnl(self) -> None:
        """오늘 누적 daily_pnl 을 DB 에서 복원.

        `_restore_state` 의 case 1/2/4 끝에서 호출 (case 3 raise 는 엔진 종료라 무관).
        get_daily_pnl 이 COALESCE(closed_at, timestamp) 쿼리라 자정 경계 case 정확 반영.
        """
        restored = await self.data_store.get_daily_pnl()
        self.account_tracker.daily_pnl = restored
        logger.info(
            "[AccountTracker] daily_pnl restored from DB: $%.2f (today=%s UTC)",
            restored,
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

    async def _restore_state(self) -> None:
        """재시작 시 잔액·포지션 복원.

        포지션 매칭 정책:
          - 거래소 O + DB O + strategy_name match:
              - active 리스트에 있으면 정상 OPEN, 없으면 ORPHAN
          - 거래소 O + DB ∅ + 전략 0개: 에러 중단
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
        self.account_tracker.set_initial_balance(initial)
        peak = await self.data_store.get_peak_equity()
        if peak > 0:
            self.account_tracker.peak_equity = peak
        self.account_tracker.update_equity(balance)

        # 포지션 매칭
        exchange_pos = await self.broker.get_position()
        open_trades = await self.data_store.get_open_trades()

        # 1) 거래소 없음 + DB 없음
        if exchange_pos is None and not open_trades:
            logger.info("Clean startup: no open position")
            await self._restore_daily_pnl()
            return

        # 2) 거래소 없음 + DB 있음 → DB의 open trades 사후 청산 처리
        # 거래소 trade history에서 실제 청산 정보 fetch 시도.
        # SL/TP 자동 청산 케이스에서 정확한 exit_price/pnl 복원. fetch 실패 시 fallback
        # (SL 가격 추정 + WARNING — 사용자가 OKX 웹에서 정확한 PnL 확인 후 수동 update 권장).
        # same-day(UTC) 청산 누적은 아래 _restore_daily_pnl 의 get_daily_pnl 쿼리가 처리.
        #   - 텔레그램 EXIT 알림 발송 안 함 (시작 시점 noise/지연 알림 혼란 회피)
        #   - update_equity 추가 호출 안 함 (위 초기 호출이 broker.get_balance
        #     ground truth 기반이라 충분, 추가 호출은 effectively no-op)
        if exchange_pos is None and open_trades:
            logger.warning(
                "DB has %d open trades but exchange has none. "
                "Attempting to fetch actual exit data from exchange...",
                len(open_trades),
            )
            now_utc = datetime.now(timezone.utc)
            for trade in open_trades:
                exit_data = await self._fetch_actual_exit(trade)
                exit_ts_ms: int | None
                if exit_data is not None:
                    actual_exit_price, actual_pnl, actual_reason, exit_ts_ms = exit_data
                    logger.info(
                        "Trade %d: 거래소에서 청산 정보 복원 — exit=%.2f pnl=%+.2f reason=%s",
                        trade["id"], actual_exit_price, actual_pnl, actual_reason,
                    )
                else:
                    # Fallback: SL/TP 가격 추정 (수수료/슬리피지 누락, timestamp 부재)
                    sl = trade.get("stop_loss")
                    tp = trade.get("take_profit")
                    fallback_price = sl if sl is not None else (tp or trade["entry_price"])
                    side_sign = 1 if trade["side"] == "long" else -1
                    actual_exit_price = fallback_price
                    actual_pnl = (
                        (fallback_price - trade["entry_price"])
                        * trade["size"] * side_sign
                    )
                    actual_reason = ExitReason.ENGINE_SHUTDOWN.value
                    exit_ts_ms = None
                    logger.warning(
                        "Trade %d: 거래소 청산 정보 fetch 실패 — SL 가격 추정 사용 "
                        "(exit=%.2f, pnl=%+.2f, 수수료/슬리피지 누락). "
                        "OKX 웹에서 정확한 PnL 확인 후 수동 update 권장.",
                        trade["id"], actual_exit_price, actual_pnl,
                    )
                pnl_pct = (
                    actual_pnl / trade["entry_price"] * 100
                    if trade["entry_price"] > 0 else 0.0
                )
                # OKX exit ts 우선, 없으면 now (fallback)
                closed_at_iso = (
                    datetime.fromtimestamp(exit_ts_ms / 1000, tz=timezone.utc).isoformat()
                    if exit_ts_ms is not None
                    else now_utc.isoformat()
                )
                await self.data_store.close_trade(
                    trade_id=trade["id"],
                    exit_price=actual_exit_price,
                    pnl=actual_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=actual_reason,
                    closed_at=closed_at_iso,
                )

            # close_trade 가 closed_at=exit_ts_iso 로 호출 → 아래 _restore_daily_pnl 의
            # get_daily_pnl 쿼리 (COALESCE 기반) 가 자동으로 same-day 만 합산.
            await self._restore_daily_pnl()
            return

        # 3) 거래소 있음 + 전략 0개 → 에러 중단
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

        # 자동 입양: active 리스트에 있으면 OPEN, 없으면 ORPHAN
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

        # 거래소 conditional order(SL/TP) 살아있는지 검증 + 누락 시 재등록
        if self.broker.is_live and self._position is not None:
            await self._verify_and_restore_sl_tp()

        # case 4 (자동 입양/orphan 포함) 모든 처리 후 daily_pnl 복원
        await self._restore_daily_pnl()

    async def _verify_and_restore_sl_tp(self) -> None:
        """거래소의 SL/TP conditional order 생존 검증 + 누락 시 재등록.

        재시작 시 거래소가 conditional order를 유지하는 게 일반적이지만 보장 X
        (사용자 수동 cancel, 거래소 정책 변경 등). 누락 시 자금 위험 노출이라
        포지션 복원 후 검증 + 재등록 권장.
        """
        if self._position is None:
            return
        sl = self._position.stop_loss
        tp = self._position.take_profit
        if sl is None and tp is None:
            logger.warning(
                "Position restored without SL/TP (orphan). "
                "Engine-level check_candle_sl_tp 미작동 — 거래소 conditional order에 의존."
            )
            return

        # SL/TP 는 OKX conditional algo order(orders-algo-pending)라
        # 일반 fetch_open_orders(orders-pending)로는 누락됨 → 살아있는데 missing
        # 오판 → 재등록 중복. broker.fetch_open_algo_orders 로 algo endpoint 조회.
        try:
            orders = await self.broker.fetch_open_algo_orders()
        except Exception as e:
            logger.warning(
                "fetch algo orders 실패 — SL/TP 검증 skip (거래소 정상 가정): %s", e
            )
            return

        sl_alive = False
        tp_alive = False
        for order in orders:
            info = order.get("info") or {}
            # OKX: algo order의 slTriggerPx/tpTriggerPx로 SL/TP 식별
            sl_trigger = info.get("slTriggerPx") or order.get("stopLossPrice")
            tp_trigger = info.get("tpTriggerPx") or order.get("takeProfitPrice")
            if sl is not None and sl_trigger:
                try:
                    if abs(float(sl_trigger) - sl) / sl < 0.001:  # 0.1% 허용
                        sl_alive = True
                except (TypeError, ValueError):
                    pass
            if tp is not None and tp_trigger:
                try:
                    if abs(float(tp_trigger) - tp) / tp < 0.001:
                        tp_alive = True
                except (TypeError, ValueError):
                    pass

        if sl is not None and not sl_alive:
            logger.warning(
                "SL conditional order missing on exchange — re-registering @ %.2f", sl
            )
            try:
                await self.broker.place_stop_loss(
                    self._position.side, sl, self._position.size,
                )
            except Exception as e:
                logger.error("SL re-registration failed: %s", e)
        if tp is not None and not tp_alive:
            logger.warning(
                "TP conditional order missing on exchange — re-registering @ %.2f", tp
            )
            try:
                await self.broker.place_take_profit(
                    self._position.side, tp, self._position.size,
                )
            except Exception as e:
                logger.error("TP re-registration failed: %s", e)
        if sl_alive and tp_alive:
            logger.info("SL/TP conditional orders verified alive on exchange")

    def _position_to_trade_dict(self) -> dict:
        """self._position을 _fetch_actual_exit 호환 trade dict로 변환."""
        pos = self._position
        return {
            "id": pos.trade_id,
            "side": pos.side.value,  # "long" 또는 "short"
            "size": pos.size,
            "entry_price": pos.entry_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "timestamp": pos.entry_time.isoformat(),
        }

    async def _sync_unexpected_close(self, last_known_price: float, now: datetime) -> None:
        """거래소가 우리 모르게 청산한 포지션 동기화.

        라이브 운영 중 다음 case에서 봉 OHLC 기반 check_candle_sl_tp가 인지 못함:
        1) SL/TP spike만 도달 (봉 OHLC 범위 밖)
        2) 사용자 manual close (OKX 웹)
        3) 거래소 강제 청산 (margin call, liquidation)

        흐름:
        - _fetch_actual_exit으로 정확한 exit_price/reason fetch
        - 실패 시 last_known_price + ENGINE_SHUTDOWN fallback
        - _close_with_funding 호출 → 정상 close 흐름 진행
          - close_position이 거래소 ∅ 인지 → 거래소 close skip
          - DB close + event publish + self._position=None
          - 텔레그램 EXIT 알림 발송
          - daily_pnl 누적
        """
        if self._position is None:
            return

        trade_dict = self._position_to_trade_dict()
        exit_data = await self._fetch_actual_exit(trade_dict)

        if exit_data is not None:
            # 운영 중이라 timestamp 미사용 (now ≈ 청산 시각, same-day 보장)
            exit_price, _, reason_str, _ = exit_data
            try:
                reason = ExitReason(reason_str)
            except ValueError:
                reason = ExitReason.SL_HIT
            logger.warning(
                "Position closed externally — syncing: exit=%.2f reason=%s "
                "(detected via exchange position ∅ at bar close)",
                exit_price, reason.value,
            )
        else:
            # Fallback: 거래소 trade history fetch 실패 시 last_known_price 사용
            exit_price = last_known_price
            reason = ExitReason.ENGINE_SHUTDOWN
            logger.warning(
                "Position closed externally but exchange trade history fetch failed. "
                "Using fallback (last_price=%.2f, reason=engine_shutdown). "
                "OKX 웹에서 정확한 exit/PnL 확인 후 수동 update 권장.",
                exit_price,
            )

        # _close_with_funding 호출 → 정상 close 흐름 (거래소 close skip + DB close + event)
        await self._close_with_funding(exit_price, reason, now)

    async def _fetch_actual_exit(
        self, trade: dict,
    ) -> tuple[float, float, str, int | None] | None:
        """거래소에서 trade의 실제 청산 정보 fetch.

        ccxt `fetch_closed_orders`로 reduceOnly + 반대 방향 closed order 찾아
        (exit_price, pnl, reason, exit_ts_ms) 반환. fetch_my_trades보다 reduceOnly
        식별이 정확 (fetch_my_trades는 reduceOnly key 누락).

        PnL = (exit - entry) × size × side_sign - 진입_fee - 청산_fee
        - 수수료는 config의 taker_fee_pct로 추정 (실제 OKX 표시값과 ~$0.5 오차 가능)

        exit_ts_ms 반환 — `_restore_state` case 2의 same-day(UTC) 판정에 사용.
        운영 중 `_sync_unexpected_close`는 미사용 (now ≈ 청산 시각이라 same-day 보장).

        Returns:
            (exit_price, pnl, reason, exit_ts_ms) 또는 None.
            exit_ts_ms: ccxt order의 timestamp (UTC ms). order에 timestamp 없으면 None.
            None 반환: paper 모드/fetch 실패 시 caller가 fallback.
        """
        if not self.broker.is_live:
            return None
        try:
            executor = getattr(self.broker, "executor", None)
            if executor is None or not hasattr(executor, "exchange"):
                return None

            from datetime import datetime
            try:
                entry_dt = datetime.fromisoformat(trade["timestamp"])
                since_ms = int(entry_dt.timestamp() * 1000)
            except Exception:
                since_ms = None

            symbol = self.config["exchange"]["symbol"]
            orders = await executor.exchange.fetch_closed_orders(
                symbol, since=since_ms, limit=50,
            )

            entry_size = float(trade["size"])
            entry_price = float(trade["entry_price"])
            entry_side_str = trade["side"]
            close_side_str = "sell" if entry_side_str == "long" else "buy"

            # reduceOnly + 반대 방향 closed order 찾기 (가장 최근부터)
            for o in reversed(orders):
                info = o.get("info") or {}
                # reduceOnly: ccxt가 raw bool로 제공하거나 info의 string ("true")으로 제공
                reduce_raw = o.get("reduceOnly")
                is_reduce = (
                    reduce_raw is True
                    or str(info.get("reduceOnly", "")).lower() == "true"
                )
                if not is_reduce:
                    continue
                if str(o.get("side", "")).lower() != close_side_str:
                    continue

                exit_price = float(o.get("average") or 0)
                if exit_price <= 0:
                    continue

                # PnL 계산: gross + 진입/청산 수수료 추정
                taker_fee = float(
                    self.config.get("accounting", {}).get("taker_fee_pct", 0.0005)
                )
                side_sign = 1 if entry_side_str == "long" else -1
                gross_pnl = (exit_price - entry_price) * entry_size * side_sign
                entry_fee_est = entry_price * entry_size * taker_fee
                close_fee_est = exit_price * entry_size * taker_fee
                net_pnl = gross_pnl - entry_fee_est - close_fee_est

                # 청산 사유 추정 (LONG: exit<entry → SL / SHORT: exit>entry → SL)
                if side_sign == 1:
                    reason = (
                        ExitReason.SL_HIT.value if exit_price < entry_price
                        else ExitReason.TP_HIT.value
                    )
                else:
                    reason = (
                        ExitReason.SL_HIT.value if exit_price > entry_price
                        else ExitReason.TP_HIT.value
                    )
                # ccxt order의 timestamp (UTC ms). caller(case 2)가 same-day 판정에 사용
                exit_ts_ms = o.get("timestamp")
                return exit_price, net_pnl, reason, exit_ts_ms
            return None
        except Exception as e:
            logger.warning("fetch_closed_orders 실패 (best-effort): %s", e)
            return None

    @staticmethod
    def _match_trade_to_exchange(
        open_trades: list[dict], exchange_pos: dict
    ) -> dict | None:
        """거래소 포지션과 DB open trade 매칭: side + size 기준.

        size tolerance 는 trade_sync.SIZE_TOLERANCE_BTC (0.005 = 0.5 contract) 로
        완화한다. DB size 는 사이징 공식 full precision (예 0.06907371),
        거래소 체결은 contract 단위 절삭 (예 0.069) 이라 구조적으로 ~7e-5 차이 →
        기존 1e-6 tolerance 로는 정상 포지션이 orphan 으로 오복원됨.
        단일 슬롯이라 side+size 로 사실상 유일 (첫 매칭 반환).
        trade_sync 의 sync 매칭과 동일 tolerance 로 일관 (DRY).
        """
        from src.live.trade_sync import SIZE_TOLERANCE_BTC
        ex_side: PositionSide = exchange_pos["side"]
        ex_size = float(exchange_pos["size"])
        for trade in open_trades:
            try:
                trade_side = PositionSide(trade["side"])
            except ValueError:
                continue
            if (
                trade_side == ex_side
                and abs(float(trade["size"]) - ex_size) <= SIZE_TOLERANCE_BTC
            ):
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

        # Circuit breaker 감시 — broker(LiveExecutor)의 cb 상태가 OPEN이면 publish
        if not self._circuit_breaker_open:
            executor = getattr(self.broker, "executor", None)
            if executor is not None:
                cb = getattr(executor, "circuit_breaker", None)
                if cb is not None and cb.is_open:
                    self._circuit_breaker_open = True
                    await self.event_bus.publish("circuit_breaker_open", {
                        "consecutive_failures": cb.consecutive_failures,
                        "threshold": cb.failure_threshold,
                    })

        # 진행 중 봉 재발행이면 전략 평가 skip
        if not self._should_process_bar(tf, ts_ms):
            return

        # master timeframe BAR_CLOSED에서만 호가창 fetch
        # — 다른 timeframe BAR_CLOSED 이벤트마다 fetch하면 중복
        # _should_process_bar 후에 위치 — ccxt가 봉 진행 중 close 변동마다
        # _on_bar_closed를 트리거하므로 같은 ts 중복 fetch 방지 필수
        if (
            self.orderbook_collector is not None
            and tf == self.master_timeframe
        ):
            try:
                self._latest_orderbook = await self.orderbook_collector.fetch_and_save()
            except Exception as e:
                logger.warning("OrderBook fetch failed: %s", e)
                self._latest_orderbook = None

        candles_slice = {t: self.data_store.get_df(t) for t in self.timeframes}
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        now = pd.to_datetime(
            candle["timestamp"], unit="ms", utc=True
        ).to_pydatetime()

        # 라이브 모드에서 거래소 포지션 상태 사전 동기화 (master_tf만).
        # 거래소가 우리 모르게 청산한 case(SL/TP spike, manual close, 강제 청산)
        # 차단. 봉 OHLC 기반 check_candle_sl_tp는 spike를 인지 못함.
        if (
            self.broker.is_live
            and tf == self.master_timeframe
            and self._position is not None
        ):
            try:
                actual = await self.broker.get_position()
            except Exception as e:
                logger.warning(
                    "get_position 실패 — 동기화 skip (거래소 정상 가정): %s", e
                )
                actual = {"placeholder": True}
            if actual is None:
                # 거래소 ∅ → 우리 모르게 청산. 동기화 후 SL/TP 캔들 검사 skip
                await self._sync_unexpected_close(close, now)

        # 청산 검사(1, 2)는 master_timeframe 봉에서만 수행.
        # 다중 TF(15m/1h/4h) 동시 마감(4h 경계 등) 시 _on_bar_closed 가 TF마다
        # 동시 실행(data_feed asyncio.gather)되는데, 청산 경로에 TF 가드가 없으면
        # 1h/4h 봉도 같은 포지션 청산을 감지 → POSITION_CLOSED 중복 발행 (EXIT
        # 알림 N개 + 메모리 daily_pnl 일시 부풀림). master_tf(가장 작은 TF=15m)가
        # 청산 판정에 가장 정밀하며, _sync_unexpected_close(master_tf 거래소 ∅ 감지)
        # + 거래소 conditional order 가 spike/외부청산을 보완한다.
        if tf == self.master_timeframe:
            # 1) SL/TP 캔들 체결 검사 (엔진 담당)
            if self._position is not None:
                fill = self.check_candle_sl_tp(self._position, high, low)
                if fill is not None:
                    exit_price, reason = fill
                    await self._close_with_funding(exit_price, reason, now)

            # 2) 전략 강제 청산 훅 (보유 중 & orphan 아님일 때만)
            if self._position is not None:
                balance = await self.broker.get_balance()
                decision = self.check_strategy_exits(
                    candles_slice, close, balance, now
                )
                if decision is not None:
                    await self._close_with_funding(close, decision.reason, now)

        # 3) 봉 마감 dispatch (entry/pyramid 평가)
        # evaluate 용 balance — 청산 후 잔액 변동 반영 위해 직전 fetch (모든 TF)
        balance = await self.broker.get_balance()
        await self.evaluate_strategies_on_bar(
            tf, candles_slice, close, balance, now
        )

        # 슬롯 차있을 때 master_tf 봉 마감마다 position 상태 로그.
        # 슬롯 비었을 때는 evaluate_strategies_on_bar 안에서 _log_signal_status 호출됨.
        if self._position is not None and tf == self.master_timeframe:
            self._log_position_status(self._position, close, now)

        # master_tf 봉 마감마다 계정 재정 상태 로그 (포지션 무관)
        if tf == self.master_timeframe:
            self._log_account_status(balance, close)

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

        # 라이브 close 직후 batch sync (best-effort, 라이브 전용).
        # paper/backtest 영향 0 — sync 함수 자체에 broker.is_live 가드
        if self.broker.is_live:
            from src.live.trade_sync import sync_all_unsynced
            try:
                result = await sync_all_unsynced(
                    broker=self.broker,
                    data_store=self.data_store,
                    symbol=self.config["exchange"]["symbol"],
                    accounting_config=self.config.get("accounting", {}),
                )
                if result["synced_count"] > 0 or result["failed_count"] > 0:
                    logger.info(
                        "Trade sync: %d synced, %d failed (errors=%d)",
                        result["synced_count"],
                        result["failed_count"],
                        len(result["errors"]),
                    )
                # synced_count > 0 시 memory daily_pnl 재정렬
                # (sync 후 DB pnl 이 OKX 실값으로 갱신됐으므로 메모리 추정값과 어긋남)
                if result["synced_count"] > 0:
                    new_daily_pnl = await self.data_store.get_daily_pnl()
                    delta = new_daily_pnl - self.account_tracker.daily_pnl
                    if abs(delta) > 0.01:
                        logger.info(
                            "[AccountTracker] daily_pnl recalibrated after sync: "
                            "$%.2f → $%.2f (Δ=%+.2f, OKX 실값 반영)",
                            self.account_tracker.daily_pnl, new_daily_pnl, delta,
                        )
                    self.account_tracker.daily_pnl = new_daily_pnl
            except Exception as e:
                logger.warning("Trade sync failed (best-effort, close 흐름 유지): %s", e)

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
        entry_order_id: str | None = None,
    ) -> int:
        return await self.data_store.log_trade(
            strategy_name=strategy_name,
            side=side.value,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order_id=entry_order_id,
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
        exit_order_id: str | None = None,
    ) -> None:
        await self.data_store.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            trading_fee=trading_fee,
            funding_fee=funding_fee,
            exit_reason=exit_reason,
            exit_order_id=exit_order_id,
            closed_at=now.isoformat(),
        )

    async def _fetch_funding_since_entry(self) -> float:
        """OKX fetch_funding_history 의 amount 를 holder net 영향 기준으로 합산.

        funding 부호를 그대로 보존한다 (양수=수익, 음수=비용).
        calc_pnl 의 `net = gross - fees + funding` 식과 일관.
        """
        if self._position is None or self._position.entry_time is None:
            return 0.0
        try:
            records = await self.broker.fetch_funding_history(
                since=self._position.entry_time.isoformat(),
            )
            return sum(float(r.get("amount", 0)) for r in records)
        except Exception as e:
            logger.warning("fetch_funding_history failed: %s", e)
            return 0.0

    # ---- 모니터링 hook override (라이브/페이퍼 INFO 출력) ----

    def _log_signal_status(self, strategy, signal, bar_context=None) -> None:
        """매 entry_tf 봉 마감 시 슬롯 비었을 때 호출 (라이브/페이퍼 INFO 출력).

        전략 비종속 generic 포맷 — side / confidence / (옵션) bar 컨텍스트만 출력한다.
        전략이 `signal.meta["note"]` (문자열) 을 채우면 그대로 덧붙인다.
        모델별 진단 정보(확률 분포·기여도 등)는 각 전략 plugin 이 자체 로깅하거나
        meta["note"] 로 요약해 전달한다.

        샘플:
          [SIGNAL] my_strategy LONG conf=0.82 → ENTRY
                   bar=80050.00 (Δ-0.12% prev) range=0.15%
        """
        meta = signal.meta or {}
        conf = signal.confidence if signal.confidence is not None else 0.0
        action_marker = " → ENTRY" if signal.is_actionable else ""

        note = meta.get("note")
        note_str = f"\n         {note}" if note else ""

        # bar 컨텍스트 — close + Δ% (prev) + range%
        bar_str = ""
        if bar_context:
            close = bar_context.get("close")
            prev = bar_context.get("prev_close")
            high = bar_context.get("high")
            low = bar_context.get("low")
            if close is not None:
                bar_str = f"\n         bar={close:.2f}"
                if prev is not None and prev > 0:
                    delta_pct = (close - prev) / prev * 100
                    bar_str += f" (Δ{delta_pct:+.2f}% prev)"
                if high is not None and low is not None and low > 0:
                    range_pct = (high - low) / low * 100
                    bar_str += f" range={range_pct:.2f}%"

        logger.info(
            "[SIGNAL] %s %s conf=%.2f%s%s%s\n",
            strategy.name, signal.side.value.upper(), conf,
            action_marker, note_str, bar_str,
        )

    def _log_position_status(self, position, current_price, now) -> None:
        """master_tf 봉 마감 시 슬롯 차있을 때 호출.

        SL/TP 가격 + 현재가 대비 Δ% 표기.
        샘플:
          [POSITION] my_strategy LONG size=0.0149 entry=67100.00 current=67235.00
                     unrealized_pnl=+$2.01 (1h32m held)
                     SL=66500.00 (-1.09% from current) TP=68000.00 (+1.14%)
        """
        from src.core.enums import PositionSide
        hold_seconds = (now - position.entry_time).total_seconds()

        if position.side == PositionSide.LONG:
            unrealized = (current_price - position.entry_price) * position.size
        else:
            unrealized = (position.entry_price - current_price) * position.size

        # SL/TP 거리 표기 (None 일 수 있음 — orphan 등)
        # \n + 11 space ([POSITION] prefix 정렬)
        sl_tp_str = ""
        if current_price > 0:
            parts = []
            if position.stop_loss is not None:
                sl_delta = (position.stop_loss - current_price) / current_price * 100
                parts.append(
                    f"SL={position.stop_loss:.2f} ({sl_delta:+.2f}% from current)"
                )
            if position.take_profit is not None:
                tp_delta = (position.take_profit - current_price) / current_price * 100
                parts.append(f"TP={position.take_profit:.2f} ({tp_delta:+.2f}%)")
            if parts:
                sl_tp_str = "\n           " + " ".join(parts)

        logger.info(
            "[POSITION] %s %s size=%.4f entry=%.2f current=%.2f"
            "\n           unrealized_pnl=%+.2f (%s held)%s\n",
            position.strategy_name, position.side.value.upper(), position.size,
            position.entry_price, current_price,
            unrealized, _fmt_hold(hold_seconds), sl_tp_str,
        )

    def _log_account_status(self, balance, current_price) -> None:
        """master_tf 봉 마감 시 계정 재정 상태(계측치) 출력 (포지션 유무 무관).

        리스크 한도/락은 모델 정책(allow_entry)이라 여기선 표기하지 않는다 —
        엔진은 계측치(잔액·equity·peak·daily_pnl·dd)만 보여준다.
        샘플 (포지션 없음, balance=$5260.46, initial=$5159.87, peak=$5320.57):
          [ACCOUNT] initial_balance=$5159.87 current_balance=$5260.46 equity=$5260.46 unrealized_pnl=+$0.00
                    total_balance_diff=+$100.59 (+1.95%)
                    daily_pnl=+$0.00
                    dd=-$60.11 (vs peak equity $5320.57)
        """
        from src.core.enums import PositionSide
        unrealized = 0.0
        if self._position is not None:
            if self._position.side == PositionSide.LONG:
                unrealized = (
                    current_price - self._position.entry_price
                ) * self._position.size
            else:
                unrealized = (
                    self._position.entry_price - current_price
                ) * self._position.size

        equity = balance + unrealized
        tracker = self.account_tracker
        initial = tracker.initial_balance
        daily_pnl = tracker.daily_pnl
        dd_abs = max(0.0, tracker.peak_equity - equity)

        # total_balance_diff — initial 대비 누적 손익
        total = balance - initial
        total_pct = (total / initial * 100) if initial > 0 else 0.0

        logger.info(
            "[ACCOUNT] initial_balance=$%.2f current_balance=$%.2f equity=$%.2f "
            "unrealized_pnl=%s"
            "\n          total_balance_diff=%s (%+.2f%%)"
            "\n          daily_pnl=%s"
            "\n          dd=-$%.2f (vs peak equity $%.2f)\n",
            initial, balance, equity, _fmt_dollar(unrealized),
            _fmt_dollar(total), total_pct,
            _fmt_dollar(daily_pnl),
            dd_abs, tracker.peak_equity,
        )
