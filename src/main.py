"""CoinBot 뼈대 CLI 엔트리포인트 — 백테스트 전용.

사용법:
  python -m src.main backtest --config config/default.yaml
                              --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.backtest.engine import BacktestEngine
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coinbot",
        description="CoinBot 뼈대 — 과거 데이터 백테스트.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest", help="과거 데이터 백테스트")
    bt.add_argument("--config", required=True, help="config YAML 경로")
    bt.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    bt.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    return parser.parse_args(argv)


def _run_backtest(config: dict[str, Any], start: str, end: str) -> None:
    engine = BacktestEngine(config, start=start, end=end)
    engine.initialize()
    engine.run()
    out_dir = engine.write_reports()
    engine.shutdown()

    print("---- Backtest Summary ----")
    for k, v in engine.summary().items():
        print(f"  {k}: {v}")
    print(f"  reports: {out_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    setup_logger(config)

    try:
        if args.command == "backtest":
            _run_backtest(config, args.start, args.end)
        else:
            logger.error("Unknown command: %s", args.command)
            return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
