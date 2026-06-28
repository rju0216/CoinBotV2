"""신호/포지션 모니터링 hook 단위 테스트.

AbstractEngine.default no-op + CoreEngine override 출력 검증.
CoreEngine 인스턴스 생성 부담 회피 위해 unbound method로 호출.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal
from src.live.engine import (
    CoreEngine,
    _build_entry_message,
    _build_exit_message,
    _fmt_hold,
)


class _StubStrategy:
    """name/params 만 가진 간단 mock — _log_signal_status 는 strategy.name 만 사용."""
    def __init__(self, name: str):
        self.name = name
        self.params = {}


class TestSignalStatusLog:
    """CoreEngine._log_signal_status — generic INFO 출력 형태 검증 (전략 비종속)."""

    def test_hold_basic(self, caplog):
        strategy = _StubStrategy("my_strategy")
        signal = Signal(side=SignalSide.HOLD, confidence=0.40)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] my_strategy HOLD" in msg
        assert "conf=0.40" in msg
        assert "→ ENTRY" not in msg  # HOLD라 actionable 아님

    def test_long_actionable_marker(self, caplog):
        strategy = _StubStrategy("my_strategy")
        signal = Signal(side=SignalSide.LONG, confidence=0.70)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] my_strategy LONG" in msg
        assert "conf=0.70" in msg
        assert "→ ENTRY" in msg

    def test_note_passthrough(self, caplog):
        """전략이 meta['note'] 를 채우면 그대로 덧붙인다."""
        strategy = _StubStrategy("my_strategy")
        signal = Signal(
            side=SignalSide.SHORT, confidence=0.65, meta={"note": "regime=trend"},
        )
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] my_strategy SHORT" in msg
        assert "→ ENTRY" in msg
        assert "regime=trend" in msg

    def test_no_meta_fallback(self, caplog):
        """meta=None 인 signal 도 안전하게 처리."""
        strategy = _StubStrategy("my_strategy")
        signal = Signal(side=SignalSide.HOLD, confidence=0.0, meta=None)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal)
        msg = caplog.records[-1].message
        assert "[SIGNAL] my_strategy HOLD" in msg
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
            strategy_name="my_strategy",
        )
        current_price = 67235.0
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, current_price, now)
        msg = caplog.records[-1].message
        assert "[POSITION] my_strategy LONG" in msg
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
            strategy_name="my_strategy",
        )
        current_price = 67500.0  # SHORT 손실
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_position_status(None, position, current_price, now)
        msg = caplog.records[-1].message
        assert "[POSITION] my_strategy SHORT" in msg
        # (67000-67500)*0.02 = -10.0
        assert "unrealized_pnl=-10.00" in msg
        assert "(0h30m held)" in msg


class TestAccountStatusLog:
    """계정 재정 상태 로그 검증."""

    def test_no_position_basic_output(self, caplog):
        """포지션 없을 때 — current_balance=equity, unrealized_pnl=0, dd=0 형식."""
        from unittest.mock import MagicMock
        from src.accounting.account_tracker import AccountTracker

        rm = AccountTracker()
        rm.set_initial_balance(1000.0)
        rm.daily_pnl = 0.0

        # MockEngine — 필요 attr만 주입
        engine = MagicMock()
        engine.account_tracker = rm
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
        """LONG 포지션 + 가격 상승 → unrealized_pnl 양수, equity 증가."""
        from unittest.mock import MagicMock
        from src.core.enums import PositionSide
        from src.core.types import Position
        from src.accounting.account_tracker import AccountTracker

        rm = AccountTracker()
        rm.set_initial_balance(1000.0)
        rm.daily_pnl = 5.30

        position = Position(
            side=PositionSide.LONG,
            size=0.05,
            entry_price=67000.0,
            entry_time=datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc),
            strategy_name="my_strategy",
        )
        engine = MagicMock()
        engine.account_tracker = rm
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
        """peak_equity 대비 dd 절대값 계산 정확성 (% 제거, $ 표기)."""
        from unittest.mock import MagicMock
        from src.accounting.account_tracker import AccountTracker

        rm = AccountTracker()
        rm.set_initial_balance(1000.0)
        rm.peak_equity = 1100.0  # 이전 peak
        rm.daily_pnl = -50.0

        engine = MagicMock()
        engine.account_tracker = rm
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
        """AbstractEngine default no-op (backtest 무영향)."""
        from src.core.engine_base import AbstractEngine
        with caplog.at_level(logging.INFO):
            result = AbstractEngine._log_account_status(None, 1000.0, 67000.0)
        assert result is None
        assert not any(
            "[ACCOUNT]" in r.message for r in caplog.records
        )


class TestSignalBarContext:
    """_log_signal_status 의 bar_context (직전 마감 봉 close/Δ%/range%) 출력 검증."""

    def test_bar_context_with_delta_and_range(self, caplog):
        """bar_context 인자가 close + Δ% + range% 출력."""
        strategy = _StubStrategy("my_strategy")
        signal = Signal(side=SignalSide.HOLD, confidence=0.92)
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
        """bar_context=None 이면 bar 출력 안 함."""
        strategy = _StubStrategy("my_strategy")
        signal = Signal(side=SignalSide.HOLD, confidence=0.40)
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_signal_status(None, strategy, signal, None)
        msg = caplog.records[-1].message
        assert "bar=" not in msg


class TestBLE71PositionSLTP:
    """_log_position_status 의 SL/TP 거리 출력 검증."""

    def test_sl_tp_distance_long_position(self, caplog):
        entry_time = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 5, 9, 14, 15, tzinfo=timezone.utc)
        position = Position(
            side=PositionSide.LONG,
            size=0.0680,
            entry_price=79687.60,
            entry_time=entry_time,
            strategy_name="my_strategy",
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


class TestAccountStatusDistances:
    """_log_account_status 의 dd 절대값 + peak equity + daily_pnl 계측 표기 검증.

    리스크 한도/락(limit/lock) 표기는 모델 정책으로 이관되어 제거됨 — 엔진 로그엔
    계측치만 남는다.
    """

    def test_account_metrics(self, caplog):
        from src.accounting.account_tracker import AccountTracker
        rm = AccountTracker()
        rm.set_initial_balance(3500.0)
        rm.peak_equity = 3500.0
        rm.daily_pnl = -50.0  # 일일 -$50

        class _Stub:
            account_tracker = rm
            _position = None
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(_Stub(), 3450.0, 80000.0)
        msg = caplog.records[-1].message
        # dd 절대값: peak-equity = 3500-3450 = 50.00
        assert "dd=-$50.00" in msg
        assert "vs peak equity $3500.00" in msg
        assert "daily_pnl=-$50.00" in msg
        # 정책 한도/락 표기는 제거됨
        assert "limit" not in msg
        assert "lock" not in msg


class TestIBLE005BarContextAndTotal:
    """_build_bar_context (직전 마감 봉 OHLC) + ACCOUNT total 라인."""

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
        """ACCOUNT 로그에 total_balance_diff + initial_balance 라인 검증.

        total 라인은 'total_balance_diff=+$X.XX (+Y.YY%)' 형식, initial 은
        첫 라인 (initial_balance=$Z.ZZ) 으로 이동.
        """
        from src.accounting.account_tracker import AccountTracker
        rm = AccountTracker()
        rm.set_initial_balance(5159.87)
        rm.peak_equity = 5320.57
        rm.daily_pnl = 0.0

        class _Stub:
            account_tracker = rm
            _position = None
        with caplog.at_level(logging.INFO, logger="src.live.engine"):
            CoreEngine._log_account_status(_Stub(), 5260.46, 73000.0)
        msg = caplog.records[-1].message
        # total = 5260.46 - 5159.87 = +100.59, $ 표기
        assert "total_balance_diff=+$100.59" in msg
        # total_pct = 100.59 / 5159.87 × 100 = +1.95%
        assert "(+1.95%)" in msg
        # initial 은 첫 라인으로 이동
        assert "initial_balance=$5159.87" in msg
        assert "current_balance=$5260.46" in msg
        # dd 라인에 peak equity 표기
        assert "vs peak equity $5320.57" in msg

    def test_fmt_dollar_helper_signed_format(self):
        """_fmt_dollar 부호+$+절대값 형식 검증."""
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


class TestBLE72TelegramMessages:
    """텔레그램 ENTRY/EXIT 알림 본문 검증 (콘솔 로그 일관 + plain text)."""

    LONG = PositionSide.LONG.value
    SHORT = PositionSide.SHORT.value

    def _pos(self, **kw):
        base = dict(
            side=PositionSide.LONG,
            size=0.0149,
            entry_price=67100.0,
            entry_time=datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc),
            strategy_name="my_strategy",
            stop_loss=66500.0,
            take_profit=68000.0,
        )
        base.update(kw)
        return Position(**base)

    # ── ENTRY ──
    def test_entry_message_includes_sl_tp_distance(self):
        msg = _build_entry_message(self._pos())
        # 1라인: side size @ entry
        assert msg.startswith(f"{self.LONG} 0.0149 @ 67100.00")
        # SL Δ% (entry 대비): (66500-67100)/67100 = -0.894%
        assert "SL=66500.00 (-0.89%)" in msg
        # TP Δ% (entry 대비): (68000-67100)/67100 = +1.341%
        assert "TP=68000.00 (+1.34%)" in msg

    def test_entry_message_sl_tp_none_omitted(self):
        """orphan 등 SL/TP None → 해당 부분 생략, 에러 없음."""
        msg = _build_entry_message(self._pos(stop_loss=None, take_profit=None))
        assert "SL=" not in msg
        assert "TP=" not in msg
        # 1라인만 (개행 없음)
        assert "\n" not in msg

    # ── EXIT ──
    def test_exit_message_full(self):
        pos = self._pos()
        closed_at = datetime(2026, 5, 9, 1, 32, tzinfo=timezone.utc)  # 1h32m held
        msg = _build_exit_message(
            pos, 13.41, exit_price=68000.0, pnl_pct=1.34, closed_at=closed_at,
        )
        # entry → exit 가격
        assert f"{self.LONG} 0.0149 @ 67100.00 → 68000.00" in msg
        # net_pnl + pnl% (부호 + $ + %)
        assert "net_pnl=+$13.41 (+1.34%)" in msg
        # 보유 시간
        assert "1h32m held" in msg

    def test_exit_message_negative_pnl_plain_text(self):
        """음수 pnl + 특수문자 plain text 안전 (Markdown 데코 없음)."""
        pos = self._pos()
        closed_at = datetime(2026, 5, 9, 0, 45, tzinfo=timezone.utc)
        msg = _build_exit_message(
            pos, -40.71, exit_price=66500.0, pnl_pct=-0.91, closed_at=closed_at,
        )
        assert "net_pnl=-$40.71 (-0.91%)" in msg
        assert "→ 66500.00" in msg
        assert "0h45m held" in msg
        # Markdown 데코 패턴 없음 (parse_mode 미사용이라 `_`/`*` 단독 문자는 안전 —
        # net_pnl 의 `_` 등 정상 텍스트. 검증 대상은 데코 패턴 `*...*` / `` `...` ``)
        assert "*" not in msg
        assert "`" not in msg

    def test_exit_message_missing_keys_graceful(self):
        """구버전 payload·orphan: exit_price/pnl_pct/closed_at None → 생략, net_pnl만."""
        msg = _build_exit_message(self._pos(), -5.0)
        assert "net_pnl=-$5.00" in msg
        # exit 가격·pnl%·hold 생략
        assert "→" not in msg
        assert "%" not in msg
        assert "held" not in msg

    # ── _fmt_hold helper ──
    def test_fmt_hold_format(self):
        assert _fmt_hold(0) == "0h00m"
        assert _fmt_hold(60) == "0h01m"
        assert _fmt_hold(3600) == "1h00m"
        assert _fmt_hold(5520) == "1h32m"      # 1h32m
        assert _fmt_hold(45 * 60) == "0h45m"

    def test_fmt_hold_negative(self):
        """음수 hold (orphan entry_time=now 등 이상 case) 정확 표기."""
        assert _fmt_hold(-6720) == "-1h52m"
        assert _fmt_hold(-450) == "-0h07m"
        assert _fmt_hold(-3600) == "-1h00m"
