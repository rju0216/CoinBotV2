"""과거 캔들 데이터 다운로드 유틸.

다중 TF + 날짜 범위 지원. run_full_backtest.bat가 병렬 백테 시작 전에 한 번 호출.

Usage:
    python scripts/download_history.py --timeframe 4h --start 2024-01-01 --end 2024-12-31
    python scripts/download_history.py --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-04-18
    python scripts/download_history.py --timeframe 4h --limit 1000   # 레거시: 최근 N개
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402


def _parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def main() -> None:
    parser = argparse.ArgumentParser(description="과거 캔들 데이터 다운로드")
    parser.add_argument(
        "--timeframe",
        default="4h",
        help="콤마 구분 TF 리스트 (예: 1d,4h,15m) 또는 단일 TF",
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="config YAML 경로"
    )
    parser.add_argument(
        "--start", help="시작일 (YYYY-MM-DD). --limit와 상호 배타."
    )
    parser.add_argument(
        "--end", help="종료일 (YYYY-MM-DD). --limit와 상호 배타."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="최근 N개 다운로드 (레거시 모드). --start/--end와 상호 배타.",
    )
    args = parser.parse_args()

    if (args.start or args.end) and args.limit:
        parser.error("--start/--end와 --limit는 동시 사용 불가")
    if not args.limit and not (args.start and args.end):
        parser.error("--start/--end 또는 --limit 중 하나는 필수")

    config = load_config(args.config)
    setup_logger(config)

    timeframes = [tf.strip() for tf in args.timeframe.split(",") if tf.strip()]
    loader = HistoricalDataLoader(config)
    try:
        for tf in timeframes:
            if args.limit:
                path = await loader.download_to_csv(tf, args.limit)
                print(f"[{tf}] saved to: {path}")
            else:
                start_ms = int(_parse_dt(args.start).timestamp() * 1000)
                end_ms = int(_parse_dt(args.end).timestamp() * 1000)
                df = await loader.download_range_merged(tf, start_ms, end_ms)
                print(f"[{tf}] merged {len(df)} candles")
    finally:
        await loader.close()


if __name__ == "__main__":
    asyncio.run(main())
