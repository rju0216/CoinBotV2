"""Live OOS monitoring (BP-2-3, 사안 D 나: 알림만).

목적: 라이브 운영 중 모델의 alpha decay 자동 감지.
- 매 봉 generate_signal 결과(LONG/SHORT)를 buffer에 record
- horizon 봉 후 실제 future return으로 actual label 산출
- 최근 window 개 (prediction, actual) 적중률이 min_acc_threshold 미만이면 알림

설계 원칙:
- 라이브 전용 (CoreEngine만 초기화). BacktestEngine은 walk-forward OOS 인프라 사용.
- in-memory buffer (재시작 시 window 빔, 재누적 후 평가 발동). DB persistence는 미래 확장.
- alert_method="log" only (telegram/email은 미래 확장).
- HOLD 신호는 buffer에 push 안 함 (방향 예측 없음).
- 다중 strategy는 strategy_name별 buffer 분리 (우리 시스템은 (C) 배타 경합이라 보통 1개).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.core.enums import SignalSide

logger = logging.getLogger(__name__)


@dataclass
class _PendingPrediction:
    strategy_name: str
    ts: datetime
    signal_side: SignalSide  # LONG / SHORT (HOLD 제외)
    entry_close: float
    target_ts: datetime  # ts + horizon * tf


@dataclass
class _EvaluatedPrediction:
    strategy_name: str
    predicted_side: SignalSide
    actual_side: SignalSide  # LONG / SHORT / HOLD
    hit: bool


class LiveOOSMonitor:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = (config.get("live", {}) or {}).get("oos_monitoring", {}) or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.window: int = int(cfg.get("window", 100))
        self.horizon: int = int(cfg.get("horizon", 10))
        self.threshold_pct: float = float(cfg.get("threshold_pct", 0.003))
        self.min_acc_threshold: float = float(cfg.get("min_acc_threshold", 0.50))
        self.cooldown_bars: int = int(cfg.get("cooldown_bars", 10))
        self.alert_method: str = str(cfg.get("alert_method", "log"))

        # entry_timeframe 문자열 → ms (record/evaluate 매칭)
        self._tf_ms = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }

        # strategy_name별 buffer
        self._pending: dict[str, list[_PendingPrediction]] = {}
        self._evaluated: dict[str, deque[_EvaluatedPrediction]] = {}

        # cooldown tracking: strategy_name → 마지막 알림 후 평가된 신호 수
        self._cooldown_counter: dict[str, int] = {}

        # BL-2-1: EventBus 통합 (CoreEngine.initialize에서 attach_event_bus)
        self._event_bus: Any | None = None

    def attach_event_bus(self, event_bus: Any) -> None:
        """CoreEngine.initialize에서 호출. _fire_alert가 OOS_DECAY publish."""
        self._event_bus = event_bus

    # ---- record / evaluate ----

    def record_prediction(
        self,
        strategy_name: str,
        entry_timeframe: str,
        ts: datetime,
        signal_side: SignalSide,
        entry_close: float,
    ) -> None:
        """generate_signal 결과 record. HOLD는 무시."""
        if not self.enabled:
            return
        if signal_side == SignalSide.HOLD:
            return
        tf_ms = self._tf_ms.get(entry_timeframe)
        if tf_ms is None:
            logger.warning("OOS monitor: unknown tf %s", entry_timeframe)
            return
        target_ts_ms = int(ts.timestamp() * 1000) + self.horizon * tf_ms
        target_ts = datetime.fromtimestamp(
            target_ts_ms / 1000, tz=ts.tzinfo
        )
        self._pending.setdefault(strategy_name, []).append(
            _PendingPrediction(
                strategy_name=strategy_name,
                ts=ts,
                signal_side=signal_side,
                entry_close=entry_close,
                target_ts=target_ts,
            )
        )

    def evaluate_pending(self, now: datetime, close: float) -> None:
        """now 시점에 horizon 도달한 pending prediction을 평가."""
        if not self.enabled:
            return
        for strategy_name, pending_list in list(self._pending.items()):
            still_pending: list[_PendingPrediction] = []
            for p in pending_list:
                if p.target_ts > now:
                    still_pending.append(p)
                    continue
                actual_side = self._classify_return(p.entry_close, close)
                hit = (p.signal_side == actual_side)
                self._record_evaluated(strategy_name, p.signal_side, actual_side, hit)
                self._check_threshold(strategy_name)
            self._pending[strategy_name] = still_pending

    def _record_evaluated(
        self,
        strategy_name: str,
        predicted: SignalSide,
        actual: SignalSide,
        hit: bool,
    ) -> None:
        buf = self._evaluated.setdefault(
            strategy_name, deque(maxlen=self.window)
        )
        buf.append(_EvaluatedPrediction(strategy_name, predicted, actual, hit))
        # cooldown counter 증가 (마지막 알림 이후 누적)
        if strategy_name in self._cooldown_counter:
            self._cooldown_counter[strategy_name] += 1

    def _classify_return(self, entry: float, future: float) -> SignalSide:
        if entry <= 0:
            return SignalSide.HOLD
        ret = (future - entry) / entry
        if ret > self.threshold_pct:
            return SignalSide.LONG
        if ret < -self.threshold_pct:
            return SignalSide.SHORT
        return SignalSide.HOLD

    # ---- 임계 체크 ----

    def _check_threshold(self, strategy_name: str) -> None:
        buf = self._evaluated.get(strategy_name)
        if buf is None or len(buf) < self.window:
            return  # window 채워지기 전엔 평가 보류
        # cooldown: 알림 발생 후 cooldown_bars 동안 재알림 차단
        cd = self._cooldown_counter.get(strategy_name)
        if cd is not None and cd < self.cooldown_bars:
            return
        acc = sum(1 for e in buf if e.hit) / len(buf)
        if acc < self.min_acc_threshold:
            self._fire_alert(strategy_name, acc)
            self._cooldown_counter[strategy_name] = 0  # cooldown 시작

    def _fire_alert(self, strategy_name: str, acc: float) -> None:
        msg = (
            f"OOS MONITOR ALERT [{strategy_name}]: accuracy {acc:.3f} < "
            f"threshold {self.min_acc_threshold:.3f} over last {self.window} signals"
        )
        logger.warning(msg)
        # BL-2-1: EventBus publish → CoreEngine subscribe → notifier (텔레그램 등)
        if self._event_bus is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._event_bus.publish(
                        "oos_decay",
                        {
                            "strategy": strategy_name,
                            "accuracy": acc,
                            "threshold": self.min_acc_threshold,
                            "window": self.window,
                        },
                    ))
            except Exception as e:
                logger.warning("OOS_DECAY event publish failed: %s", e)

    # ---- BL-2 추가 step (DD''=가): 라이브 시작 직전 warm-up ----

    def warmup_from_history(
        self,
        strategy_name: str,
        entry_timeframe: str,
        bars: Any,  # pd.DataFrame (timestamp index, ohlcv 컬럼)
        signal_iter: Any,  # Callable[[ts], SignalSide] — plugin.generate_signal 시뮬
        cutoff_dt: datetime,
        learned_oos_acc: float | None = None,
    ) -> dict[str, Any]:
        """학습 cutoff 이후 historical bars로 buffer 사전 채움.

        라이브 시작 직전 호출하면 OOS monitor가 즉시 적중률 보유 + alpha decay
        사전 감지 가능. 학습 OOS Acc(예: 0.75)와 현재 적중률 비교로 격차 정량.

        Args:
            strategy_name: 모델 이름 (record_prediction과 일관)
            entry_timeframe: 모델 entry_tf (BAR_CLOSED 시점 매칭용)
            bars: cutoff_dt 이후 entry_tf 봉 (DataFrame, ts index, ohlcv)
            signal_iter: ts → SignalSide 함수 (plugin.generate_signal 결과 시뮬)
            cutoff_dt: 학습 cutoff datetime. 이 이후 봉만 처리
            learned_oos_acc: 학습 시 OOS Acc (train_meta.oos_accuracy). 격차 비교용

        Returns:
            {samples, accuracy, gap, learned_oos_acc} 통계 dict
        """
        if not self.enabled or bars is None or len(bars) == 0:
            return {
                "samples": 0,
                "accuracy": None,
                "gap": None,
                "learned_oos_acc": learned_oos_acc,
            }

        for ts in bars.index:
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if ts_dt <= cutoff_dt:
                continue
            try:
                side = signal_iter(ts_dt)
            except Exception as e:
                logger.warning("warmup signal_iter 실패 ts=%s: %s", ts_dt, e)
                continue
            close = float(bars.loc[ts, "close"])
            # HOLD는 record_prediction에서 자동 skip
            self.record_prediction(
                strategy_name, entry_timeframe, ts_dt, side, close,
            )
            # 매 봉 evaluate (라이브 BAR_CLOSED 패턴 동일)
            self.evaluate_pending(ts_dt, close)

        samples = self.get_window_size(strategy_name)
        accuracy = self.get_accuracy(strategy_name)
        gap = None
        if accuracy is not None and learned_oos_acc is not None:
            gap = float(learned_oos_acc) - float(accuracy)
        return {
            "samples": samples,
            "accuracy": accuracy,
            "gap": gap,
            "learned_oos_acc": learned_oos_acc,
        }

    # ---- 외부 조회용 (테스트 + 디버깅) ----

    def get_accuracy(self, strategy_name: str) -> float | None:
        buf = self._evaluated.get(strategy_name)
        if buf is None or len(buf) == 0:
            return None
        return sum(1 for e in buf if e.hit) / len(buf)

    def get_window_size(self, strategy_name: str) -> int:
        buf = self._evaluated.get(strategy_name)
        return len(buf) if buf is not None else 0
