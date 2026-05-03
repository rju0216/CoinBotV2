"""LiveOOSMonitor 단위 테스트 (BP-2-3)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.core.enums import SignalSide
from src.live.oos_monitor import LiveOOSMonitor


def _cfg(**overrides) -> dict:
    base = {
        "live": {
            "oos_monitoring": {
                "enabled": True,
                "window": 5,
                "horizon": 2,
                "threshold_pct": 0.003,
                "min_acc_threshold": 0.5,
                "cooldown_bars": 10,
                "alert_method": "log",
            }
        }
    }
    base["live"]["oos_monitoring"].update(overrides)
    return base


def _ts(minutes_offset: int) -> datetime:
    return datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minutes_offset
    )


class TestDisabledMonitor:
    def test_disabled_record_noop(self):
        cfg = _cfg(enabled=False)
        m = LiveOOSMonitor(cfg)
        m.record_prediction("s1", "15m", _ts(0), SignalSide.LONG, 67000.0)
        assert m.get_window_size("s1") == 0


class TestRecord:
    def test_hold_signal_skipped(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.HOLD, 67000.0)
        m.evaluate_pending(_ts(60), 67500.0)  # 60분 후 (horizon 2 * 15m = 30분 도달)
        assert m.get_window_size("s1") == 0

    def test_long_signal_recorded_and_pending(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.LONG, 67000.0)
        # horizon 도달 전 (15분만 경과; horizon=2 → 30분)
        m.evaluate_pending(_ts(15), 67100.0)
        assert m.get_window_size("s1") == 0  # 아직 평가 안 됨


class TestEvaluation:
    def test_long_hit_when_future_up(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.LONG, 67000.0)
        # horizon 30분 도달, 0.5% 상승 → actual=LONG → hit
        m.evaluate_pending(_ts(30), 67000.0 * 1.005)
        assert m.get_window_size("s1") == 1
        assert m.get_accuracy("s1") == 1.0

    def test_long_miss_when_future_flat(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.LONG, 67000.0)
        # 30분 후 0.1% 변화 (threshold 0.3% 미만) → actual=HOLD → predicted LONG miss
        m.evaluate_pending(_ts(30), 67000.0 * 1.001)
        assert m.get_window_size("s1") == 1
        assert m.get_accuracy("s1") == 0.0

    def test_short_hit_when_future_down(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.SHORT, 67000.0)
        m.evaluate_pending(_ts(30), 67000.0 * 0.99)  # -1% → SHORT
        assert m.get_accuracy("s1") == 1.0


class TestWindow:
    def test_window_caps_at_max(self):
        cfg = _cfg(window=3)
        m = LiveOOSMonitor(cfg)
        for i in range(10):
            m.record_prediction("s1", "15m", _ts(i * 15), SignalSide.LONG, 67000.0)
        # 모든 entry에 대해 horizon 도달
        m.evaluate_pending(_ts(1000), 67000.0 * 1.01)  # +1% → LONG hit
        # window=3 → 마지막 3개만 유지
        assert m.get_window_size("s1") == 3
        assert m.get_accuracy("s1") == 1.0

    def test_per_strategy_isolation(self):
        m = LiveOOSMonitor(_cfg())
        m.record_prediction("s1", "15m", _ts(0), SignalSide.LONG, 67000.0)
        m.record_prediction("s2", "15m", _ts(0), SignalSide.SHORT, 67000.0)
        m.evaluate_pending(_ts(30), 67000.0 * 1.005)  # +0.5% → LONG
        # s1: hit (LONG 예측 → LONG actual)
        # s2: miss (SHORT 예측 → LONG actual)
        assert m.get_accuracy("s1") == 1.0
        assert m.get_accuracy("s2") == 0.0


class TestThresholdAlert:
    def test_alert_fires_when_acc_below_threshold(self, caplog):
        cfg = _cfg(window=4, min_acc_threshold=0.5)
        m = LiveOOSMonitor(cfg)
        # 4개 SHORT 예측, 모두 actual LONG → 0/4 적중률
        for i in range(4):
            m.record_prediction("s1", "15m", _ts(i * 15), SignalSide.SHORT, 67000.0)
        with caplog.at_level(logging.WARNING):
            m.evaluate_pending(_ts(1000), 67000.0 * 1.01)
        assert m.get_window_size("s1") == 4
        assert m.get_accuracy("s1") == 0.0
        assert any("OOS MONITOR ALERT" in r.message for r in caplog.records)

    def test_no_alert_when_window_not_full(self, caplog):
        cfg = _cfg(window=10, min_acc_threshold=0.99)
        m = LiveOOSMonitor(cfg)
        for i in range(3):
            m.record_prediction("s1", "15m", _ts(i * 15), SignalSide.SHORT, 67000.0)
        with caplog.at_level(logging.WARNING):
            m.evaluate_pending(_ts(1000), 67000.0 * 1.01)
        # window=10 < 3개 평가됐어도 임계 체크 안 됨
        assert not any("OOS MONITOR ALERT" in r.message for r in caplog.records)

    def test_cooldown_blocks_re_alert(self, caplog):
        cfg = _cfg(window=2, min_acc_threshold=0.5, cooldown_bars=5)
        m = LiveOOSMonitor(cfg)
        # 첫 알림 트리거 (2 SHORT all miss)
        for i in range(2):
            m.record_prediction("s1", "15m", _ts(i * 15), SignalSide.SHORT, 67000.0)
        with caplog.at_level(logging.WARNING):
            m.evaluate_pending(_ts(100), 67000.0 * 1.01)
            alerts_first = sum(
                1 for r in caplog.records if "OOS MONITOR ALERT" in r.message
            )
        # cooldown 동안 추가 신호 (alert 추가 발생 안 해야 함)
        for i in range(3):
            m.record_prediction(
                "s1", "15m", _ts(200 + i * 15), SignalSide.SHORT, 67000.0
            )
        with caplog.at_level(logging.WARNING):
            m.evaluate_pending(_ts(500), 67000.0 * 1.01)
            alerts_after = sum(
                1 for r in caplog.records if "OOS MONITOR ALERT" in r.message
            )
        assert alerts_first == 1
        assert alerts_after == alerts_first  # cooldown으로 추가 알림 차단


class TestUnknownTimeframe:
    def test_unknown_tf_skipped(self, caplog):
        m = LiveOOSMonitor(_cfg())
        with caplog.at_level(logging.WARNING):
            m.record_prediction(
                "s1", "30m", _ts(0), SignalSide.LONG, 67000.0  # 30m 미정의
            )
        assert m.get_window_size("s1") == 0
