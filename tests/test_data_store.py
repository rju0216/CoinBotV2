"""DataStore 단위 테스트 — closed_at 컬럼 + get_daily_pnl COALESCE 쿼리 (I-BLE001).

실제 SQLite DB (tmp_path 격리) 로 schema/쿼리 검증.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pytest

from src.data.store import DataStore


def _make_config(db_path: str) -> dict:
    return {
        "database": {"path": db_path},
        "paper": {"initial_balance": 10000},
    }


@pytest.mark.asyncio
async def test_close_trade_persists_closed_at(tmp_path):
    """I-BLE001: close_trade(closed_at=...) 호출 후 SELECT 결과 일치."""
    db_path = os.path.join(tmp_path, "test_coinbot.db")
    store = DataStore(_make_config(db_path), mode="paper")
    await store.initialize()
    try:
        # trade 생성
        trade_id = await store.log_trade(
            strategy_name="ensemble",
            side="long",
            size=0.075,
            entry_price=82000.0,
            stop_loss=81500.0,
            take_profit=83000.0,
        )
        # close with explicit closed_at
        closed_at_iso = "2026-05-07T03:15:42.123456+00:00"
        await store.close_trade(
            trade_id=trade_id,
            exit_price=81700.0,
            pnl=-22.5,
            pnl_pct=-0.027,
            trading_fee=6.14,
            funding_fee=0.0,
            exit_reason="sl_hit",
            exit_order_id="okx_exit_123",
            closed_at=closed_at_iso,
        )
        # SELECT 검증
        cursor = await store._db.execute(
            "SELECT exit_price, pnl, closed_at, exit_order_id FROM trades WHERE id=?",
            (trade_id,),
        )
        row = await cursor.fetchone()
        assert row[0] == pytest.approx(81700.0)
        assert row[1] == pytest.approx(-22.5)
        assert row[2] == closed_at_iso
        assert row[3] == "okx_exit_123"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_close_trade_default_closed_at_uses_now(tmp_path):
    """I-BLE001: close_trade(closed_at=None) → datetime.now(timezone.utc) fallback."""
    db_path = os.path.join(tmp_path, "test_coinbot.db")
    store = DataStore(_make_config(db_path), mode="paper")
    await store.initialize()
    try:
        trade_id = await store.log_trade(
            strategy_name="ensemble", side="long", size=0.075,
            entry_price=82000.0, stop_loss=81500.0, take_profit=83000.0,
        )
        before = datetime.now(timezone.utc)
        await store.close_trade(
            trade_id=trade_id, exit_price=81700.0, pnl=-22.5, pnl_pct=-0.027,
            # closed_at 미지정 → fallback
        )
        after = datetime.now(timezone.utc)
        cursor = await store._db.execute(
            "SELECT closed_at FROM trades WHERE id=?", (trade_id,),
        )
        row = await cursor.fetchone()
        closed_at_dt = datetime.fromisoformat(row[0])
        assert before <= closed_at_dt <= after
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_daily_pnl_uses_coalesce_closed_at(tmp_path):
    """I-BLE001 ② 핵심: 자정 경계 case — 어제 open + 오늘 close 가 오늘 daily_pnl 에 합산.
    open 기준 (기존) 으로는 어제로 계산됐을 영역이 closed_at 기준으로 정확 반영.
    """
    db_path = os.path.join(tmp_path, "test_coinbot.db")
    store = DataStore(_make_config(db_path), mode="paper")
    await store.initialize()
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        # Trade A: 어제 open + 어제 close — daily_pnl 합산 제외
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status, closed_at) VALUES (?, 'e', 'long', 0.1, 80000, "
            "80100, ?, 'closed', ?)",
            (f"{yesterday_str}T20:00:00+00:00", 10.0, f"{yesterday_str}T22:00:00+00:00"),
        )
        # Trade B: 어제 open + 오늘 close (자정 경계 case) — daily_pnl 합산 포함 (closed_at 기준)
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status, closed_at) VALUES (?, 'e', 'short', 0.1, 80000, "
            "79800, ?, 'closed', ?)",
            (f"{yesterday_str}T23:30:00+00:00", 20.0, f"{today_str}T02:30:00+00:00"),
        )
        # Trade C: 오늘 open + 오늘 close — 정상 합산
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status, closed_at) VALUES (?, 'e', 'long', 0.1, 80000, "
            "80050, ?, 'closed', ?)",
            (f"{today_str}T05:00:00+00:00", 5.0, f"{today_str}T07:00:00+00:00"),
        )
        # Trade D: closed_at NULL (fallback to timestamp via COALESCE), open 오늘 → 합산
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status) VALUES (?, 'e', 'long', 0.1, 80000, 80030, ?, 'closed')",
            (f"{today_str}T08:00:00+00:00", 3.0),
        )
        await store._db.commit()

        daily_pnl = await store.get_daily_pnl()
        # 합산 영역: Trade B (20.0) + Trade C (5.0) + Trade D (3.0) = 28.0
        # Trade A (어제 closed_at) 는 제외
        assert daily_pnl == pytest.approx(28.0)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_daily_pnl_coalesce_fallback_to_timestamp(tmp_path):
    """I-BLE001: closed_at NULL 인 기존 trade 는 COALESCE 로 timestamp 사용 (legacy 호환)."""
    db_path = os.path.join(tmp_path, "test_coinbot.db")
    store = DataStore(_make_config(db_path), mode="paper")
    await store.initialize()
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        # 어제 open + closed_at NULL → COALESCE fallback (어제) → 합산 제외
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status) VALUES (?, 'e', 'long', 0.1, 80000, 80100, 10.0, 'closed')",
            (f"{yesterday_str}T12:00:00+00:00",),
        )
        # 오늘 open + closed_at NULL → COALESCE fallback (오늘) → 합산 포함
        await store._db.execute(
            "INSERT INTO trades (timestamp, strategy_name, side, size, entry_price, "
            "exit_price, pnl, status) VALUES (?, 'e', 'long', 0.1, 80000, 80050, 5.0, 'closed')",
            (f"{today_str}T08:00:00+00:00",),
        )
        await store._db.commit()

        daily_pnl = await store.get_daily_pnl()
        assert daily_pnl == pytest.approx(5.0)
    finally:
        await store.close()
