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
        # BLE-7-1 보강: conf class label (argmax=H, 0.40)
        assert "conf=H:0.40" in msg
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
        """포지션 없을 때 — current_balance=equity, unrealized_pnl=0, dd=0 (I-BLE006 형식)."""
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
        assert "initial_balance=$1000.00" in msg
        assert "current_balance=$1000.00" in msg
        assert "equity=$1000.00" in msg
        assert "unrealized_pnl=+$0.00" in msg
        assert "daily_pnl=+$0.00" in msg
        # peak=initial=1000 → dd_abs=0 → 표기 -$0.00
        assert "dd=-$0.00" in msg
        # peak equity 표기
        assert "vs peak equity $1000.00" in msg

    def test_long_position_unrealized_profit(self, caplog):
        """LONG 포지션 + 가격 상승 → unrealized_pnl 양수, equity 증가 (I-BLE006 형식)."""
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
        assert "unrealized_pnl=+$5.00" in msg
        # equity = 1000 + 5 = 1005.00
        assert "equity=$1005.00" in msg
        assert "daily_pnl=+$5.30" in msg

    def test_drawdown_pct_calculation(self, caplog):
        """peak_equity 대비 dd 절대값 계산 정확성 (I-BLE006: % 제거, $ 표기)."""
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
        # dd = peak - equity = 1100 - 990 = 110.00 (음수 텍스트로 표기)
        assert "dd=-$110.00" in msg
        assert "current_balance=$990.00" in msg
        assert "daily_pnl=-$50.00" in msg
        # vs peak equity 표기
        assert "vs peak equity $1100.00" in msg


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


class TestBLE71SignalEnhancement:
    """BLE-7-1: _log_signal_status 의 sub_probs + bar_context 출력 검증."""

    def test_sub_probs_full_format(self, caplog):
        """ensemble sub_probs 풀 [S:H:L] 표기 확인."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.92,
            meta={
                "probs": [0.05, 0.92, 0.03],
                "contributors": ["ml_lightgbm", "ml_xgboost"],
                "sub_probs": {
                    "ml_lightgbm": [0.04, 0.93, 0.03],
                    "ml_xgboost": [0.06, 0.91, 0.03],
                },
            },
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "sub_probs={" in msg
        assert "ml_lightgbm:[S:0.04 H:0.93 L:0.03]" in msg
        assert "ml_xgboost:[S:0.06 H:0.91 L:0.03]" in msg

    def test_bar_context_with_delta_and_range(self, caplog):
        """bar_context 인자가 close + Δ% + range% 출력."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.92,
            meta={"probs": [0.05, 0.92, 0.03]},
        )
        bar_context = {
            "close": 80050.0, "prev_close": 80150.0, "high": 80100.0, "low": 79980.0,
        }
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal, bar_context)
        msg = caplog.records[-1].message
        assert "bar=80050.00" in msg
        # Δ = (80050-80150)/80150 = -0.1247%
        assert "(Δ-0.12% prev)" in msg
        # range = (80100-79980)/79980 = 0.150%
        assert "range=0.15%" in msg

    def test_bar_context_none_no_bar_str(self, caplog):
        """bar_context=None 이면 bar 출력 안 함 (backward-compat)."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.40,
            meta={"probs": [0.31, 0.40, 0.29]},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal, None)
        msg = caplog.records[-1].message
        assert "bar=" not in msg


class TestBLE71PositionSLTP:
    """BLE-7-1: _log_position_status 의 SL/TP 거리 출력 검증."""

    def test_sl_tp_distance_long_position(self, caplog):
        entry_time = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 5, 9, 14, 15, tzinfo=timezone.utc)
        position = Position(
            side=PositionSide.LONG,
            size=0.0680,
            entry_price=79687.60,
            entry_time=entry_time,
            strategy_name="ensemble",
            stop_loss=79100.00,
            take_profit=80800.00,
        )
        current_price = 80050.00
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, current_price, now)
        msg = caplog.records[-1].message
        # SL 거리: (79100-80050)/80050 = -1.187%
        assert "SL=79100.00 (-1.19% from current)" in msg
        # TP 거리: (80800-80050)/80050 = +0.937%
        assert "TP=80800.00 (+0.94%)" in msg

    def test_sl_tp_none_no_distance_str(self, caplog):
        """orphan 등 SL/TP 없으면 SL/TP 출력 안 함."""
        entry_time = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 5, 9, 14, 15, tzinfo=timezone.utc)
        position = Position(
            side=PositionSide.LONG,
            size=0.05, entry_price=80000.0, entry_time=entry_time,
            strategy_name="_unknown",
            stop_loss=None, take_profit=None,
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, 80100.0, now)
        msg = caplog.records[-1].message
        assert "SL=" not in msg
        assert "TP=" not in msg


class TestBLE71ConfClassLabel:
    """BLE-7-1 보강: conf class label (S/H/L) 표기 검증.

    signal.side 와 별개로 probs argmax 의 class 표기. threshold 미달 시
    signal.side=HOLD 라도 conf 가 다른 class(L/S) 의 점수일 수 있음 — 사용자
    혼란 영역의 본질 검증.
    """

    def test_conf_class_hold_argmax(self, caplog):
        """probs argmax=H 인 정상 case — conf=H:0.92."""
        strategy = _StubStrategy("ensemble", threshold=0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.92,
            meta={"probs": [0.05, 0.92, 0.03]},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "conf=H:0.92" in msg

    def test_conf_class_long_argmax_threshold_below(self, caplog):
        """probs argmax=L 인데 threshold 미달 → signal=HOLD + conf=L:0.55
        (사용자 혼란 본질: signal.side ≠ argmax class)."""
        strategy = _StubStrategy("ensemble", threshold=0.60)
        # probs argmax=L (0.55), 하지만 threshold 미달이라 signal=HOLD
        # confidence 는 argmax class 의 점수 (0.55)
        signal = Signal(
            side=SignalSide.HOLD,
            confidence=0.55,
            meta={"probs": [0.05, 0.40, 0.55]},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        # signal.side=HOLD 가 메시지 앞에 표기
        assert "[SIGNAL] ensemble HOLD" in msg
        # conf 는 argmax class (L) 의 점수로 표기 — 핵심
        assert "conf=L:0.55" in msg

    def test_conf_no_probs_fallback(self, caplog):
        """probs 없으면 (단일 plugin meta None 등) class label 없이 fallback."""
        strategy = _StubStrategy("ml_lightgbm")
        signal = Signal(side=SignalSide.HOLD, confidence=0.0, meta=None)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        # probs 없으면 conf=0.00 형태 (class label 없음, backward-compat)
        assert "conf=0.00" in msg
        # class label prefix 형식은 없어야 함
        assert "conf=H:" not in msg
        assert "conf=L:" not in msg
        assert "conf=S:" not in msg


class TestBLE71AccountRiskDistance:
    """BLE-7-1 / I-BLE006: _log_account_status 의 daily 한도/DD 락 거리 + peak equity 검증."""

    def test_account_risk_distances(self, caplog):
        from src.risk.manager import RiskManager
        rm = RiskManager({"risk": {
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.35,
            "max_position_size_btc": 1.0,
            "max_concurrent_positions": 1,
        }})
        rm.set_initial_balance(3500.0)
        rm.peak_equity = 3500.0
        rm.daily_pnl = -50.0  # 일일 -$50

        # _log_account_status 직접 호출 — self.risk_manager + self._position 만 사용
        class _Stub:
            risk_manager = rm
            _position = None
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(_Stub(), 3450.0, 80000.0)
        msg = caplog.records[-1].message
        # I-BLE006: daily 한도 = config max_daily_loss_pct (5%) + balance×5%
        # current_balance=3450 × 0.05 = $172.50
        assert "limit -5%" in msg
        assert "-$172.50" in msg
        # DD 락 한도: 3500 × 0.35 = $1225.00
        assert "lock -35%" in msg
        assert "-$1225.00" in msg
        # I-BLE006: dd 절대값만 표기 ($), % 제거. peak-equity = 3500-3450 = 50.00
        assert "dd=-$50.00" in msg
        # I-BLE006: peak equity 표기
        assert "vs peak equity $3500.00" in msg
        # daily_pnl I-BLE006 형식
        assert "daily_pnl=-$50.00" in msg


class TestIBLE005BarContextAndTotal:
    """I-BLE005: _build_bar_context (직전 마감 봉 OHLC) + ACCOUNT total 라인."""

    def test_build_bar_context_default_backtest_uses_iloc_minus_one(self):
        """engine_base default (LAST_CLOSED_BAR_IDX=-1) — 백테 영역.
        df = ts 미만 슬라이스라 iloc[-1] 이 이미 직전 마감 봉.
        """
        import pandas as pd
        from src.core.engine_base import AbstractEngine

        df = pd.DataFrame(
            [
                {"open": 80000, "high": 80200, "low": 79900, "close": 80100},   # 그 전 봉
                {"open": 80100, "high": 80300, "low": 80000, "close": 80250},   # 직전 마감 봉 (iloc[-1])
            ]
        )
        # AbstractEngine 직접 호출 — current_price 인자 미사용
        ctx = AbstractEngine._build_bar_context(AbstractEngine, df, current_price=99999.0)
        # 직전 마감 봉 (iloc[-1]) 의 OHLC
        assert ctx["close"] == pytest.approx(80250.0)
        assert ctx["high"] == pytest.approx(80300.0)
        assert ctx["low"] == pytest.approx(80000.0)
        # prev_close = iloc[-2]["close"]
        assert ctx["prev_close"] == pytest.approx(80100.0)

    def test_build_bar_context_core_engine_uses_iloc_minus_two(self):
        """CoreEngine override (LAST_CLOSED_BAR_IDX=-2) — 라이브 영역.
        df 의 iloc[-1] 은 새 봉 single tick, iloc[-2] 가 직전 마감 봉.
        """
        import pandas as pd
        from src.live.engine import CoreEngine

        df = pd.DataFrame(
            [
                {"open": 80000, "high": 80200, "low": 79900, "close": 80100},   # 그 전 봉
                {"open": 80100, "high": 80300, "low": 80000, "close": 80250},   # 직전 마감 봉 (iloc[-2])
                {"open": 80250, "high": 80250, "low": 80250, "close": 80250},   # 새 봉 single tick (iloc[-1])
            ]
        )
        ctx = CoreEngine._build_bar_context(CoreEngine, df, current_price=99999.0)
        # iloc[-2] 의 진정한 OHLC
        assert ctx["close"] == pytest.approx(80250.0)
        assert ctx["high"] == pytest.approx(80300.0)
        assert ctx["low"] == pytest.approx(80000.0)
        # range = (80300-80000)/80000 = 0.375% — 0 아님 (single tick 영역 회피)
        # prev_close = iloc[-3]["close"]
        assert ctx["prev_close"] == pytest.approx(80100.0)

    def test_build_bar_context_returns_none_when_df_too_short(self):
        """edge case: df=None / len(df) < abs(LAST_CLOSED_BAR_IDX) → None."""
        import pandas as pd
        from src.core.engine_base import AbstractEngine
        from src.live.engine import CoreEngine

        # df=None
        assert AbstractEngine._build_bar_context(AbstractEngine, None, 80000.0) is None
        assert CoreEngine._build_bar_context(CoreEngine, None, 80000.0) is None
        # default LAST=-1, len(df)=0 → None
        assert AbstractEngine._build_bar_context(
            AbstractEngine, pd.DataFrame(), 80000.0,
        ) is None
        # CoreEngine LAST=-2, len(df)=1 → None (abs(-2)=2 미만)
        df_one = pd.DataFrame([{"open": 0, "high": 0, "low": 0, "close": 80000}])
        assert CoreEngine._build_bar_context(CoreEngine, df_one, 80000.0) is None

    def test_account_log_includes_total_vs_initial(self, caplog):
        """I-BLE005 / I-BLE006: ACCOUNT 로그에 total_balance_diff + initial_balance 라인 영역.

        I-BLE006: total 라인은 'total_balance_diff=+$X.XX (+Y.YY%)' 형식, initial 영역은
        첫 라인 (initial_balance=$Z.ZZ) 으로 이동.
        """
        from src.risk.manager import RiskManager
        rm = RiskManager({"risk": {"max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.35}})
        rm.set_initial_balance(5159.87)
        rm.peak_equity = 5320.57
        rm.daily_pnl = 0.0

        class _Stub:
            risk_manager = rm
            _position = None
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(_Stub(), 5260.46, 73000.0)
        msg = caplog.records[-1].message
        # I-BLE006: total = 5260.46 - 5159.87 = +100.59, $ 표기
        assert "total_balance_diff=+$100.59" in msg
        # total_pct = 100.59 / 5159.87 × 100 = +1.95%
        assert "(+1.95%)" in msg
        # I-BLE006: initial 은 첫 라인으로 이동
        assert "initial_balance=$5159.87" in msg
        assert "current_balance=$5260.46" in msg
        # I-BLE006: dd 영역에 peak equity 표기
        assert "vs peak equity $5320.57" in msg

    def test_fmt_dollar_helper_signed_format(self):
        """I-BLE006: _fmt_dollar 부호+$+절대값 형식 검증."""
        from src.live.engine import _fmt_dollar
        assert _fmt_dollar(0.0) == "+$0.00"
        assert _fmt_dollar(100.59) == "+$100.59"
        assert _fmt_dollar(-5.30) == "-$5.30"
        # -0.0 → +$0.00 (Python: -0.0 >= 0 True)
        assert _fmt_dollar(-0.0) == "+$0.00"
        # 큰 음수
        assert _fmt_dollar(-1234.56) == "-$1234.56"
        # 소수 영역 round
        assert _fmt_dollar(1.005) == "+$1.00" or _fmt_dollar(1.005) == "+$1.01"
