"""신호/포지션 모니터링 hook 단위 테스트 (BL-2-3 hotfix-E).

AbstractEngine.default no-op + CoreEngine override 출력 검증.
CoreEngine 인스턴스 생성 부담 회피 위해 unbound method로 호출.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal
from src.live.engine import CoreEngine


class _StubStrategy:
    """params만 가진 간단 mock — _log_signal_status는 strategy.name/params만 사용."""
    def __init__(self, name: str, threshold: float = 0.55):
        self.name = name
        self.params = {"confidence_threshold": threshold}


class TestSignalStatusLog:
    """CoreEngine._log_signal_status — INFO 출력 형태 검증."""

    def test_ensemble_hold_with_probs_and_contributors(self, caplog):
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.40,
            meta={
                "probs": [0.31, 0.40, 0.29],
                "contributors": ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer"],
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ensemble HOLD" in msg
        assert "probs=[S:0.31 H:0.40 L:0.29]" in msg
        assert "conf=0.40" in msg
        assert "threshold=0.55" in msg
        assert "contributors=" in msg
        assert "→ ENTRY" not in msg  # HOLD라 actionable 아님

    def test_long_actionable_marker(self, caplog):
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.LONG,
            confidence=0.70,
            meta={"probs": [0.10, 0.20, 0.70]},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ensemble LONG" in msg
        assert "→ ENTRY" in msg

    def test_single_model_no_contributors(self, caplog):
        """단일 모델 — meta에 contributors 없으면 출력 안 함."""
        strategy = _StubStrategy("dl_transformer", threshold=0.60)
        signal = Signal(
            side=SignalSide.SHORT,
            confidence=0.65,
            meta={"probs": [0.65, 0.20, 0.15]},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] dl_transformer SHORT" in msg
        assert "threshold=0.60" in msg
        assert "→ ENTRY" in msg
        assert "contributors=" not in msg

    def test_no_meta_fallback(self, caplog):
        """meta=None인 signal도 안전하게 처리."""
        strategy = _StubStrategy("ml_lightgbm")
        signal = Signal(side=SignalSide.HOLD, confidence=0.0, meta=None)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ml_lightgbm HOLD" in msg
        assert "probs=" not in msg  # meta 없으면 probs_str 비어있음
        assert "conf=0.00" in msg


class TestPositionStatusLog:
    """CoreEngine._log_position_status — INFO 출력 형태 검증."""

    def test_long_position_unrealized_pnl_positive(self, caplog):
        entry_time = datetime(2026, 5, 5, 21, 30, tzinfo=timezone.utc)
        now = datetime(2026, 5, 5, 23, 5, tzinfo=timezone.utc)  # 1h35m 후
        position = Position(
            side=PositionSide.LONG,
            size=0.0149,
            entry_price=67100.0,
            entry_time=entry_time,
            strategy_name="ensemble",
        )
        current_price = 67235.0
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, current_price, now)
        msg = caplog.records[-1].message
        assert "[POSITION] ensemble LONG" in msg
        assert "size=0.0149" in msg
        assert "entry=67100.00" in msg
        assert "current=67235.00" in msg
        # (67235-67100)*0.0149 = 2.0115
        assert "unrealized_pnl=+2.01" in msg
        assert "(1h35m held)" in msg

    def test_short_position_unrealized_pnl_negative(self, caplog):
        entry_time = datetime(2026, 5, 5, 21, 0, tzinfo=timezone.utc)
        now = datetime(2026, 5, 5, 21, 30, tzinfo=timezone.utc)  # 0h30m 후
        position = Position(
            side=PositionSide.SHORT,
            size=0.02,
            entry_price=67000.0,
            entry_time=entry_time,
            strategy_name="dl_transformer",
        )
        current_price = 67500.0  # SHORT 손실
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, current_price, now)
        msg = caplog.records[-1].message
        assert "[POSITION] dl_transformer SHORT" in msg
        # (67000-67500)*0.02 = -10.0
        assert "unrealized_pnl=-10.00" in msg
        assert "(0h30m held)" in msg


class TestSignalStatusFailReason:
    """I-BL007 Phase 1: 추론 실패 case 출력 검증."""

    def test_ensemble_unavailable_subs_output(self, caplog):
        """ensemble이 unavailable_subs meta(dict 형태)로 HOLD 반환 시 (no inference: ...) 출력."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            meta={
                "unavailable_subs": {
                    "ml_lightgbm": {"reason": "nan_in_last_row"},
                    "dl_lstm": {"reason": "dropna_lt_lookback"},
                }
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ensemble HOLD" in msg
        assert "no inference:" in msg
        assert "ml_lightgbm=nan_in_last_row" in msg
        assert "dl_lstm=dropna_lt_lookback" in msg
        assert "threshold=0.55" in msg
        # 정상 path 출력은 없어야 함
        assert "probs=" not in msg
        assert "conf=" not in msg

    def test_single_model_fail_reason_output(self, caplog):
        """단일 plugin이 fail_reason meta로 HOLD 반환 시 (no inference: ...) 출력."""
        strategy = _StubStrategy("dl_transformer", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            meta={
                "fail_reason": "dropna_lt_lookback",
                "available_rows": 45,
                "required_lookback": 60,
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] dl_transformer HOLD" in msg
        assert "no inference: dl_transformer=dropna_lt_lookback" in msg
        assert "probs=" not in msg

    def test_partial_unavailable_with_contributors(self, caplog):
        """ensemble probs_list < min_models — unavailable_subs 우선 출력."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            meta={
                "unavailable_subs": {
                    "dl_lstm": {"reason": "dropna_lt_lookback"},
                    "dl_transformer": {"reason": "dropna_lt_lookback"},
                },
                "contributors": ["ml_lightgbm"],
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "no inference:" in msg
        assert "dl_lstm=dropna_lt_lookback" in msg
        assert "dl_transformer=dropna_lt_lookback" in msg


class TestSignalStatusDiagnostic:
    """I-BL007 Phase 3-C: 진단 정보 출력 검증."""

    def test_normal_signal_with_gap(self, caplog):
        """정상 case + gap > 0 (진행 중 봉 영향) → diag 정보 추가 출력."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.97,
            meta={
                "probs": [0.01, 0.97, 0.02],
                "contributors": ["ml_lightgbm", "ml_xgboost"],
                "gap_to_latest": 1,
                "used_row_ts": "2026-05-06 04:30:00+00:00",
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ensemble HOLD" in msg
        assert "probs=" in msg
        assert "(gap=1, used_ts=2026-05-06 04:30:00+00:00)" in msg

    def test_normal_signal_without_gap(self, caplog):
        """정상 case + gap=0 (가장 최근 봉 사용) → diag 정보 미출력 (noise 회피)."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.97,
            meta={
                "probs": [0.01, 0.97, 0.02],
                "contributors": ["ml_lightgbm", "ml_xgboost"],
                "gap_to_latest": 0,
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] ensemble HOLD" in msg
        assert "(gap=" not in msg

    def test_failure_with_nan_by_tf(self, caplog):
        """실패 case + nan_by_tf → timeframe별 NaN 컬럼 출력."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            meta={
                "unavailable_subs": {
                    "ml_lightgbm": {
                        "reason": "all_features_nan",
                        "nan_by_tf": {
                            "1h": ["body_ratio", "upper_shadow"],
                            "4h": ["atr_pct"],
                        },
                    },
                    "dl_lstm": {
                        "reason": "dropna_lt_lookback",
                        "available_rows": 45,
                        "required_lookback": 60,
                    },
                },
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "no inference:" in msg
        assert "ml_lightgbm=all_features_nan" in msg
        assert "1h: body_ratio,upper_shadow" in msg
        assert "4h: atr_pct" in msg
        assert "dl_lstm=dropna_lt_lookback" in msg
        assert "available:45/60" in msg


class TestAccountStatusLog:
    """BL-2-4 hotfix-G: 계정 재정 상태 로그 검증."""

    def test_no_position_basic_output(self, caplog):
        """포지션 없을 때 — balance=equity, unrealized=0, dd=0."""
        from unittest.mock import MagicMock
        from src.risk.manager import RiskManager

        rm = RiskManager({"risk": {"max_daily_loss_pct": 0.05}})
        rm.set_initial_balance(1000.0)
        rm.daily_pnl = 0.0

        # MockEngine — 필요 attr만 주입
        engine = MagicMock()
        engine.risk_manager = rm
        engine._position = None

        CoreEngine._log_account_status(engine, 1000.0, 67000.0)

        # 마지막 INFO 로그 확인 (caplog는 자동으로 src.live.engine logger 캡처)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(engine, 1000.0, 67000.0)
        msg = caplog.records[-1].message
        assert "[ACCOUNT]" in msg
        assert "balance=$1000.00" in msg
        assert "equity=$1000.00" in msg
        assert "unrealized=+0.00" in msg
        assert "daily_pnl=+0.00" in msg
        assert "dd=0.00%" in msg

    def test_long_position_unrealized_profit(self, caplog):
        """LONG 포지션 + 가격 상승 → unrealized 양수, equity 증가."""
        from unittest.mock import MagicMock
        from src.core.enums import PositionSide
        from src.core.types import Position
        from src.risk.manager import RiskManager

        rm = RiskManager({"risk": {"max_daily_loss_pct": 0.05}})
        rm.set_initial_balance(1000.0)
        rm.daily_pnl = 5.30

        position = Position(
            side=PositionSide.LONG,
            size=0.05,
            entry_price=67000.0,
            entry_time=datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc),
            strategy_name="ensemble",
        )
        engine = MagicMock()
        engine.risk_manager = rm
        engine._position = position

        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(engine, 1000.0, 67100.0)
        msg = caplog.records[-1].message
        # (67100-67000) × 0.05 = 5.00
        assert "unrealized=+5.00" in msg
        # equity = 1000 + 5 = 1005.00
        assert "equity=$1005.00" in msg
        assert "daily_pnl=+5.30" in msg

    def test_drawdown_pct_calculation(self, caplog):
        """peak_equity 대비 dd% 계산 정확성."""
        from unittest.mock import MagicMock
        from src.risk.manager import RiskManager

        rm = RiskManager({"risk": {"max_daily_loss_pct": 0.05}})
        rm.set_initial_balance(1000.0)
        rm.peak_equity = 1100.0  # 이전 peak
        rm.daily_pnl = -50.0

        engine = MagicMock()
        engine.risk_manager = rm
        engine._position = None

        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(engine, 990.0, 67000.0)
        msg = caplog.records[-1].message
        # dd = (1100 - 990) / 1100 = 10.00%
        assert "dd=10.00%" in msg
        assert "balance=$990.00" in msg
        assert "daily_pnl=-50.00" in msg


class TestAbstractEngineDefaultNoOp:
    """AbstractEngine default hook이 no-op이라 backtest에 영향 없음 검증."""

    def test_default_log_signal_status_returns_none(self, caplog):
        from src.core.engine_base import AbstractEngine
        strategy = _StubStrategy("any")
        signal = Signal(side=SignalSide.HOLD)
        with caplog.at_level(logging.INFO):
            result = AbstractEngine._log_signal_status(None, strategy, signal)
        assert result is None
        # default no-op이라 INFO 출력 0
        assert not any(
            "[SIGNAL]" in r.message for r in caplog.records
        )

    def test_default_log_position_status_returns_none(self, caplog):
        from src.core.engine_base import AbstractEngine
        position = Position(
            side=PositionSide.LONG,
            size=0.01,
            entry_price=67000.0,
            entry_time=datetime(2026, 5, 5, tzinfo=timezone.utc),
            strategy_name="any",
        )
        with caplog.at_level(logging.INFO):
            result = AbstractEngine._log_position_status(
                None, position, 67100.0, datetime(2026, 5, 5, 1, tzinfo=timezone.utc)
            )
        assert result is None
        assert not any(
            "[POSITION]" in r.message for r in caplog.records
        )

    def test_default_log_account_status_returns_none(self, caplog):
        """BL-2-4 hotfix-G: AbstractEngine default no-op (backtest 무영향)."""
        from src.core.engine_base import AbstractEngine
        with caplog.at_level(logging.INFO):
            result = AbstractEngine._log_account_status(None, 1000.0, 67000.0)
        assert result is None
        assert not any(
            "[ACCOUNT]" in r.message for r in caplog.records
        )
