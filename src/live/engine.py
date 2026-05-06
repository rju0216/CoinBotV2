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
from src.live.oos_monitor import LiveOOSMonitor
from src.utils.notifier import Notifier, build_notifier_from_config

logger = logging.getLogger(__name__)


def _format_failure_detail(
    strategy_name: str,
    unavailable_subs: dict | None,
    fail_reason: str | None,
    meta: dict,
) -> str:
    """추론 실패 case 상세 포맷 (I-BL007 Phase 3-C)."""
    def _format_sub(name: str, info: dict) -> str:
        reason = info.get("reason", "unknown")
        parts = [f"{name}={reason}"]
        nan_by_tf = info.get("nan_by_tf")
        if nan_by_tf:
            tf_strs = [
                f"{tf}: {','.join(cols)}"
                for tf, cols in nan_by_tf.items()
            ]
            parts.append("{" + "; ".join(tf_strs) + "}")
        avail = info.get("available_rows")
        req = info.get("required_lookback")
        if avail is not None and req is not None:
            parts.append(f"{{available:{avail}/{req}}}")
        return " ".join(parts)

    if unavailable_subs:
        return ", ".join(
            _format_sub(name, info)
            for name, info in unavailable_subs.items()
        )
    info = {
        "reason": fail_reason,
        "nan_by_tf": meta.get("nan_by_tf"),
        "available_rows": meta.get("available_rows"),
        "required_lookback": meta.get("required_lookback"),
    }
    return _format_sub(strategy_name, info)


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
            # BL-2-1: OOS monitor가 EventBus publish 가능하도록 attach
            self.oos_monitor.attach_event_bus(self.event_bus)
            logger.info(
                "OOS monitor enabled: window=%d, horizon=%d, threshold=%.3f",
                self.oos_monitor.window,
                self.oos_monitor.horizon,
                self.oos_monitor.min_acc_threshold,
            )

        # BL-2-1: notifier 인프라 초기화 + EventBus subscribe
        self.notifier: Notifier = build_notifier_from_config(self.config)
        self.risk_manager.attach_event_bus(self.event_bus)
        self._setup_notifier_subscriptions()
        # Circuit breaker 발동 시 새 진입 차단 (사안 U''=나)
        self._circuit_breaker_open: bool = False

        # BL-2-2: 호가창 collector 초기화 (config.live.orderbook.enabled=true 시)
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

        # BL-2 추가 step (DD''=가): OOS monitor warm-up
        # 학습 cutoff 이후 historical candles로 buffer 사전 채움 → 라이브 시작 즉시 적중률 보유
        if self.oos_monitor is not None:
            try:
                await self._warmup_oos_monitor()
            except Exception as e:
                logger.warning("OOS warmup failed (non-blocking): %s", e)

    async def _warmup_oos_monitor(self) -> None:
        """학습 cutoff 이후 historical candles로 OOS monitor buffer 사전 채움.

        I-BL003 fix: train_meta 추출은 strategy.extract_train_meta()로 위임.
        단일 모델은 default impl이 model_path → train_meta.json 처리. ensemble은
        sub-plugin 집계 (cutoff=min, acc=mean) override 사용. paper 운영 중
        record_prediction되는 buffer key가 strategy.name이므로 active strategy
        자체를 warm-up해야 buffer가 활용됨.
        """
        from src.data.historical import HistoricalDataLoader
        loader = HistoricalDataLoader(self.config)
        try:
            for strategy in self.strategies:
                try:
                    await self._warmup_one_strategy(strategy, loader)
                except Exception as e:
                    logger.warning(
                        "OOS warmup [%s] failed: %s", strategy.name, e,
                    )
        finally:
            await loader.close()

    async def _warmup_one_strategy(self, strategy, loader) -> None:
        """단일 strategy warm-up — train_meta로 cutoff/learned_acc 추출 + 시뮬."""
        # 1. train_meta 추출 (I-BL003 fix: strategy.extract_train_meta로 위임)
        cutoff_dt, learned_acc = strategy.extract_train_meta()
        if cutoff_dt is None:
            logger.warning(
                "OOS warmup [%s]: train cutoff 추출 실패 — skip",
                strategy.name,
            )
            return

        # 2. cutoff_dt 이후 ~ 현재까지 historical candles 다운로드 (캐시 활용)
        from datetime import datetime, timezone
        end_dt = datetime.now(timezone.utc)
        # warmup용 인디케이터 history_bars 추가 (entry_tf max indicator window 고려)
        history_bars = int(self.config.get("data", {}).get("history_bars", 300))
        entry_tf = strategy.entry_timeframe

        candles_per_tf: dict = {}
        for tf in strategy.required_timeframes:
            from src.data.historical import TF_MS
            tf_ms = TF_MS.get(tf, 60_000)
            start_ms = int(cutoff_dt.timestamp() * 1000) - history_bars * tf_ms
            end_ms = int(end_dt.timestamp() * 1000)
            df = await loader.download_range_merged(tf, start_ms, end_ms)
            candles_per_tf[tf] = df

        if entry_tf not in candles_per_tf or candles_per_tf[entry_tf].empty:
            logger.warning("OOS warmup [%s]: entry_tf 데이터 없음 — skip", strategy.name)
            return

        # 3. signal_iter 정의 — ts마다 ctx 빌드 + plugin.generate_signal
        master_df = candles_per_tf[entry_tf]
        import pandas as pd

        # I-BL004 fix: warmup용 features 1회 사전계산 후 self._features_cache에 임시 주입.
        # _build_ctx가 cache 자동 lookup → plugin.generate_signal이 매 ts마다 81 피처를
        # 처음부터 재계산(O(N²))하던 것을 1회로 축소. BacktestEngine._build_features_cache
        # 와 동일 패턴 (DRY). warmup 종료 시 finally에서 cache 비움 → 라이브 entry path
        # 무영향 (라이브는 매 봉 재계산이 default 의도).
        from src.strategy.features import compute_multi_tf_features
        try:
            self._features_cache[entry_tf] = compute_multi_tf_features(
                candles_per_tf, entry_tf,
            )
            logger.info(
                "OOS warmup [%s]: features cache built (entry_tf=%s, rows=%d)",
                strategy.name, entry_tf, len(self._features_cache[entry_tf]),
            )
        except Exception as e:
            logger.warning(
                "OOS warmup [%s]: features cache build 실패 — fallback to per-bar compute: %s",
                strategy.name, e,
            )

        try:
            def signal_iter(ts_dt):
                ts = pd.Timestamp(ts_dt).tz_convert("UTC") if pd.Timestamp(ts_dt).tz else pd.Timestamp(ts_dt, tz="UTC")
                # ts 직전까지 slice (lookahead 차단, I-B007 패턴)
                slice_dict = {
                    tf: df[df.index < ts] for tf, df in candles_per_tf.items()
                }
                # current_price = open of ts
                try:
                    current_price = float(master_df.loc[ts, "open"])
                except KeyError:
                    return None  # ts 미존재
                ctx = self._build_ctx(strategy, slice_dict, current_price, 10000.0, ts_dt)
                try:
                    signal = strategy.generate_signal(ctx)
                    return signal.side
                except Exception as e:
                    logger.debug("warmup signal_iter [%s] ts=%s 실패: %s", strategy.name, ts_dt, e)
                    return None

            # 4. monitor에 warm-up 위임. cutoff 이후 entry_tf 봉만 처리
            # (signal_iter가 None 반환하면 record_prediction 시 SignalSide(None) 오류 → 사전 필터)
            from src.core.enums import SignalSide

            def safe_signal_iter(ts_dt):
                side = signal_iter(ts_dt)
                return side if side is not None else SignalSide.HOLD

            result = self.oos_monitor.warmup_from_history(
                strategy_name=strategy.name,
                entry_timeframe=entry_tf,
                bars=master_df,
                signal_iter=safe_signal_iter,
                cutoff_dt=cutoff_dt,
                learned_oos_acc=learned_acc,
            )

            # 5. 결과 로그 + 격차 알림 (EE''=yes)
            logger.info(
                "OOS warmup [%s] complete: samples=%d, accuracy=%s, learned_oos_acc=%s, gap=%s",
                strategy.name, result["samples"],
                f"{result['accuracy']:.4f}" if result["accuracy"] is not None else None,
                f"{result['learned_oos_acc']:.4f}" if result["learned_oos_acc"] is not None else None,
                f"{result['gap']:+.4f}" if result["gap"] is not None else None,
            )
            # 격차 임계 도달 시 oos_decay publish (EE''=yes)
            decay_threshold = float(
                (self.config.get("live", {}) or {}).get("oos_monitoring", {}).get(
                    "warmup_decay_threshold_pct", 0.10,
                )
            )
            if result["gap"] is not None and result["gap"] >= decay_threshold:
                await self.event_bus.publish("oos_decay", {
                    "strategy": strategy.name,
                    "accuracy": result["accuracy"],
                    "threshold": self.oos_monitor.min_acc_threshold,
                    "learned_oos_acc": result["learned_oos_acc"],
                    "gap": result["gap"],
                    "warmup_samples": result["samples"],
                    "source": "warmup",
                })
        finally:
            # I-BL004 fix: warmup 종료 시 cache 비움 — 라이브 entry path가 stale 데이터
            # 사용 안 하도록 (cache가 cutoff 시점까지만 포함, 라이브 새 봉 미반영).
            self._features_cache.pop(entry_tf, None)

    def _setup_notifier_subscriptions(self) -> None:
        """BL-2-1: 주요 EventType → notifier 송신 라우팅.

        levels config로 각 이벤트의 송신 활성/비활성 결정 (V'' 사용자 결정 반영).
        """
        notif_cfg = (self.config.get("live", {}) or {}).get(
            "notifications", {}
        ) or {}
        levels = notif_cfg.get("levels", {}) or {}

        async def _on_drawdown_locked(data):
            if not levels.get("drawdown_lock", True):
                return
            await self.notifier.send(
                "ERROR",
                "Drawdown lock triggered",
                f"DD {data.get('drawdown_pct', 0):.2f}% >= "
                f"{data.get('max_drawdown_pct', 0):.2f}%. Trading halted.",
                **data,
            )

        async def _on_daily_loss_locked(data):
            if not levels.get("daily_loss_lock", True):
                return
            await self.notifier.send(
                "ERROR",
                "Daily loss limit reached",
                f"PnL ${data.get('daily_pnl', 0):.2f} <= ${data.get('limit', 0):.2f}",
                **data,
            )

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

        async def _on_oos_decay(data):
            if not levels.get("oos_decay", True):
                return
            await self.notifier.send(
                "WARNING",
                f"OOS decay [{data.get('strategy', 'unknown')}]",
                f"accuracy {data.get('accuracy', 0):.3f} < "
                f"threshold {data.get('threshold', 0):.3f}",
                **data,
            )

        async def _on_position_opened(pos):
            if not levels.get("position_open", True):  # V'' 사용자 결정으로 default true
                return
            await self.notifier.send(
                "INFO",
                f"ENTRY [{pos.strategy_name}]",
                f"{pos.side.value} {pos.size:.4f} @ {pos.entry_price:.2f}",
                strategy=pos.strategy_name,
                side=pos.side.value,
                size=pos.size,
                entry_price=pos.entry_price,
            )

        async def _on_position_closed(data):
            if not levels.get("position_close", True):  # V'' 사용자 결정으로 default true
                return
            pos = data.get("position")
            pnl = data.get("pnl", 0)
            reason = data.get("reason", "")
            if pos is None:
                return
            await self.notifier.send(
                "INFO",
                f"EXIT [{pos.strategy_name}] {reason}",
                f"net_pnl=${pnl:.2f}",
                strategy=pos.strategy_name,
                pnl=pnl,
                reason=reason,
            )

        self.event_bus.subscribe("drawdown_locked", _on_drawdown_locked)
        self.event_bus.subscribe("daily_loss_locked", _on_daily_loss_locked)
        self.event_bus.subscribe("circuit_breaker_open", _on_circuit_breaker)
        self.event_bus.subscribe("oos_decay", _on_oos_decay)
        self.event_bus.subscribe(EventType.POSITION_OPENED.value, _on_position_opened)
        self.event_bus.subscribe(EventType.POSITION_CLOSED.value, _on_position_closed)

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

        # BL-2-1: Circuit breaker 감시 — broker(LiveExecutor)의 cb 상태가 OPEN이면 publish
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

        # 진행 중 봉 재발행이면 전략 평가 skip (I-005)
        if not self._should_process_bar(tf, ts_ms):
            return

        # BL-2-2: master timeframe BAR_CLOSED에서만 호가창 fetch (Y''=가)
        # — 다른 timeframe BAR_CLOSED 이벤트마다 fetch하면 중복
        # I-BL005 fix: _should_process_bar 후로 이동 — ccxt가 봉 진행 중 close 변동마다
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

        # BL-2-3 hotfix-E: 슬롯 차있을 때 master_tf 봉 마감마다 position 상태 로그.
        # 슬롯 비었을 때는 evaluate_strategies_on_bar 안에서 _log_signal_status 호출됨.
        if self._position is not None and tf == self.master_timeframe:
            self._log_position_status(self._position, close, now)

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

    # ---- BL-2-3 hotfix-E: 모니터링 hook override (라이브/페이퍼 INFO 출력) ----

    def _log_signal_status(self, strategy, signal) -> None:
        """매 entry_tf 봉 마감 시 슬롯 비었을 때 호출.

        정상 inference 샘플 (dropped=0, 깔끔):
          [SIGNAL] ensemble HOLD probs=[S:0.31 H:0.40 L:0.29] conf=0.40 threshold=0.55
                   contributors=[ml_lightgbm, ml_xgboost, dl_lstm, dl_transformer]

        I-BL007 Phase 3-C: 정상 + dropped > 0 (진행 중 봉 영향 잔존):
          [SIGNAL] ensemble HOLD probs=[...] conf=... threshold=0.55 contributors=[...]
                   (dropped=1, used_ts=2026-05-06 04:30:00)

        I-BL007 Phase 3-C: 추론 실패 + 진단 정보:
          [SIGNAL] ensemble HOLD (no inference: ml_lightgbm=all_features_nan
                   {1h: body_ratio,upper_shadow; 4h: atr_pct},
                   dl_lstm=dropna_lt_lookback {available:45/60}) threshold=0.55
        """
        meta = signal.meta or {}
        probs = meta.get("probs")
        contributors = meta.get("contributors")
        threshold = float(strategy.params.get("confidence_threshold", 0.55))
        conf = signal.confidence if signal.confidence is not None else 0.0

        # 추론 실패 case
        unavailable_subs = meta.get("unavailable_subs")
        fail_reason = meta.get("fail_reason")
        if unavailable_subs or (fail_reason and not probs):
            detail = _format_failure_detail(
                strategy.name, unavailable_subs, fail_reason, meta,
            )
            logger.info(
                "[SIGNAL] %s %s (no inference: %s) threshold=%.2f",
                strategy.name, signal.side.value.upper(), detail, threshold,
            )
            return

        # 정상 case
        probs_str = ""
        if probs and len(probs) == 3:
            probs_str = (
                f" probs=[S:{probs[0]:.2f} H:{probs[1]:.2f} L:{probs[2]:.2f}]"
            )
        action_marker = " → ENTRY" if signal.is_actionable else ""
        contrib_str = (
            f" contributors={contributors}" if contributors else ""
        )

        # I-BL007 Phase 3-C: gap > 0인 경우만 진단 정보 추가 (noise 최소화)
        # gap = 가장 최근 봉 대비 사용된 row까지의 봉 수. 0=정상, N+=진행 중 봉 영향.
        diag_str = ""
        gap = meta.get("gap_to_latest", 0)
        if gap and gap > 0:
            used_ts = meta.get("used_row_ts")
            if used_ts is not None:
                diag_str = f" (gap={gap}, used_ts={used_ts})"
            else:
                diag_str = f" (gap={gap})"

        logger.info(
            "[SIGNAL] %s %s%s conf=%.2f threshold=%.2f%s%s%s",
            strategy.name, signal.side.value.upper(), probs_str, conf, threshold,
            action_marker, contrib_str, diag_str,
        )


    def _log_position_status(self, position, current_price, now) -> None:
        """master_tf 봉 마감 시 슬롯 차있을 때 호출.

        샘플:
          [POSITION] ensemble LONG size=0.0149 entry=67100.00 current=67235.00
                     unrealized_pnl=+$2.01 (1h32m held)
        """
        from src.core.enums import PositionSide
        hold_seconds = (now - position.entry_time).total_seconds()
        hold_h = int(hold_seconds // 3600)
        hold_m = int((hold_seconds % 3600) // 60)

        if position.side == PositionSide.LONG:
            unrealized = (current_price - position.entry_price) * position.size
        else:
            unrealized = (position.entry_price - current_price) * position.size

        logger.info(
            "[POSITION] %s %s size=%.4f entry=%.2f current=%.2f "
            "unrealized_pnl=%+.2f (%dh%02dm held)",
            position.strategy_name, position.side.value.upper(), position.size,
            position.entry_price, current_price, unrealized,
            hold_h, hold_m,
        )
