"""AccountTracker 단위 테스트 (tracking 전용 — 정책 enforcement 없음)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.accounting.account_tracker import AccountTracker


def test_set_initial_balance_sets_peak():
    t = AccountTracker()
    t.set_initial_balance(1000.0)
    assert t.initial_balance == 1000.0
    assert t.peak_equity == 1000.0


def test_update_equity_raises_peak_only():
    t = AccountTracker()
    t.set_initial_balance(1000.0)
    t.update_equity(1200.0)
    assert t.peak_equity == 1200.0
    t.update_equity(1100.0)  # 하락은 peak 안 낮춤
    assert t.peak_equity == 1200.0


def test_add_pnl_accumulates():
    t = AccountTracker()
    t.add_pnl(10.0)
    t.add_pnl(-3.0)
    assert t.daily_pnl == 7.0


def test_drawdown_pct():
    t = AccountTracker()
    t.set_initial_balance(1000.0)
    t.update_equity(1200.0)  # peak=1200
    assert t.drawdown_pct(1200.0) == 0.0
    assert t.drawdown_pct(1080.0) == 0.1  # (1200-1080)/1200
    # equity가 peak보다 높으면 0 (음수 방지)
    assert t.drawdown_pct(1300.0) == 0.0


def test_maybe_reset_for_new_day():
    t = AccountTracker()
    d1 = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
    d2 = datetime(2024, 1, 2, 0, 30, tzinfo=timezone.utc)
    # 첫 호출: base date 설정만
    assert t.maybe_reset_for_new_day(d1) is False
    t.add_pnl(-50.0)
    # 같은 날: no-op
    assert t.maybe_reset_for_new_day(d1) is False
    assert t.daily_pnl == -50.0
    # 다음 날: reset
    assert t.maybe_reset_for_new_day(d2) is True
    assert t.daily_pnl == 0.0
