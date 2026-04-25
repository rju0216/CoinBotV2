"""SQLite 기반 거래·자산 기록 저장소.

owner 분기와 exit_plan partial 트레이드 흐름은 제거. trades 테이블에
strategy_name 컬럼을 도입하여 전략별 추적과 단계 7의 자동 입양 기반을 마련.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import pandas as pd

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        """
        Args:
            config: 통합 config dict.
            mode: "live" | "paper" | "backtest". CLI subcommand에서 결정되어 주입.
                  DB 파일명에 접미사로 부여 (예: coinbot_live.db).
        """
        base_path = config.get("database", {}).get("path", "data/coinbot.db")
        stem, ext = os.path.splitext(base_path)
        self.db_path = f"{stem}_{mode}{ext}"
        self.mode = mode
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_pct REAL,
                trading_fee REAL DEFAULT 0,
                funding_fee REAL DEFAULT 0,
                exit_reason TEXT,
                status TEXT DEFAULT 'open'
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                total_equity REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                snapshot TEXT NOT NULL,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        await self._db.commit()
        logger.info("DataStore initialized: %s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ---- 초기 잔액 / 메타 ----

    async def get_initial_balance(self) -> float | None:
        cursor = await self._db.execute(
            "SELECT value FROM bot_meta WHERE key='initial_balance'"
        )
        row = await cursor.fetchone()
        return float(row[0]) if row else None

    async def set_initial_balance(self, balance: float) -> None:
        existing = await self.get_initial_balance()
        if existing is not None:
            return
        await self._db.execute(
            "INSERT INTO bot_meta (key, value) VALUES ('initial_balance', ?)",
            (str(balance),),
        )
        await self._db.commit()

    # ---- in-memory DataFrame 캐시 (TF별 캔들) ----

    def set_dataframe(self, timeframe: str, df: pd.DataFrame) -> None:
        self._dataframes[timeframe] = df.copy()

    def append_candle(self, timeframe: str, candle: dict[str, Any]) -> None:
        if timeframe not in self._dataframes:
            self._dataframes[timeframe] = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        ts = pd.to_datetime(candle["timestamp"], unit="ms", utc=True)
        row = pd.DataFrame(
            [{
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
            }],
            index=pd.DatetimeIndex([ts], name="timestamp"),
        )
        df = self._dataframes[timeframe]
        if ts in df.index:
            df.loc[ts] = row.iloc[0]
        else:
            self._dataframes[timeframe] = pd.concat([df, row])

    def get_df(self, timeframe: str) -> pd.DataFrame:
        return self._dataframes.get(timeframe, pd.DataFrame())

    # ---- 거래 기록 ----

    async def log_trade(
        self,
        strategy_name: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """INSERT INTO trades
               (timestamp, strategy_name, side, size, entry_price,
                stop_loss, take_profit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open')""",
            (now, strategy_name, side, size, entry_price, stop_loss, take_profit),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        trading_fee: float = 0.0,
        funding_fee: float = 0.0,
        exit_reason: str | None = None,
    ) -> None:
        await self._db.execute(
            """UPDATE trades SET exit_price=?, pnl=?, pnl_pct=?, trading_fee=?,
               funding_fee=?, exit_reason=?, status='closed' WHERE id=?""",
            (exit_price, pnl, pnl_pct, trading_fee, funding_fee, exit_reason, trade_id),
        )
        await self._db.commit()

    async def get_trades(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM trades WHERE status=? ORDER BY id DESC", (status,)
            )
        else:
            cursor = await self._db.execute("SELECT * FROM trades ORDER BY id DESC")
        columns = [desc[0] for desc in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    async def get_open_trades(self) -> list[dict[str, Any]]:
        return await self.get_trades(status="open")

    async def get_daily_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades "
            "WHERE status='closed' AND timestamp LIKE ?",
            (f"{today}%",),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0

    # ---- 자산 기록 ----

    async def log_equity(
        self,
        balance: float,
        unrealized_pnl: float = 0.0,
        exchange_equity: float | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        total = (
            exchange_equity if exchange_equity is not None else balance + unrealized_pnl
        )
        await self._db.execute(
            "INSERT INTO equity (timestamp, balance, unrealized_pnl, total_equity) "
            "VALUES (?, ?, ?, ?)",
            (now, balance, unrealized_pnl, total),
        )
        await self._db.commit()

    async def get_last_balance(self) -> float | None:
        cursor = await self._db.execute(
            "SELECT balance FROM equity ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_peak_equity(self) -> float:
        cursor = await self._db.execute("SELECT MAX(total_equity) FROM equity")
        row = await cursor.fetchone()
        return row[0] if row and row[0] else 0.0

    async def get_equity_history(self) -> pd.DataFrame:
        cursor = await self._db.execute(
            "SELECT timestamp, total_equity FROM equity ORDER BY timestamp"
        )
        rows = await cursor.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "total_equity"])
        df = pd.DataFrame(rows, columns=["timestamp", "total_equity"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    # ---- 진입/청산 시점 지표 스냅샷 ----

    async def save_trade_snapshot(
        self, trade_id: int, event: str, snapshot: dict[str, Any]
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO trade_snapshots (trade_id, event, timestamp, snapshot) "
            "VALUES (?, ?, ?, ?)",
            (trade_id, event, now, json.dumps(snapshot)),
        )
        await self._db.commit()

    async def get_trade_snapshots(self, trade_id: int) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT event, timestamp, snapshot FROM trade_snapshots "
            "WHERE trade_id=? ORDER BY id",
            (trade_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"event": event, "timestamp": ts, "snapshot": json.loads(snap)}
            for event, ts, snap in rows
        ]
