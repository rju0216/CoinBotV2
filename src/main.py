"""CoinBot 뼈대 프로토타입 CLI 엔트리포인트.

사용법:
  python -m src.main paper    --config config/default.yaml
  python -m src.main live     --config config/default.yaml
  python -m src.main backtest --config config/default.yaml
                              --start 2024-01-01 --end 2024-12-31

mode 필드는 config에 두지 않고 subcommand로 결정 (정책 6 (c)).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from src.backtest.engine import BacktestEngine
from src.live.engine import CoreEngine
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coinbot",
        description=(
            "CoinBot 뼈대 프로토타입. subcommand로 실행 모드를 선택."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd, help_text in (
        ("paper", "페이퍼 모드 실시간 시뮬레이션"),
        ("live", "라이브 모드 OKX 실거래"),
    ):
        sp = sub.add_parser(cmd, help=help_text)
        sp.add_argument("--config", required=True, help="config YAML 경로")

    bt = sub.add_parser("backtest", help="과거 데이터 백테스트")
    bt.add_argument("--config", required=True, help="config YAML 경로")
    bt.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    bt.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")

    return parser.parse_args(argv)


async def _run_live_or_paper(config: dict[str, Any], mode: str) -> None:
    engine = CoreEngine(config, mode=mode)
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, ValueError):
            # Windows 또는 non-main thread에서는 add_signal_handler 미지원
            pass

    try:
        await engine.initialize()
        run_task = asyncio.create_task(engine.run())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            [run_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        await engine.shutdown()


async def _run_backtest(
    config: dict[str, Any], start: str, end: str, config_path: str
) -> None:
    engine = BacktestEngine(config, start=start, end=end)
    out_dir = None
    try:
        await engine.initialize()
        await engine.run()
        result = await engine.get_result()
        out_dir = engine.write_reports(config_path=config_path)
    finally:
        await engine.shutdown()

    print("---- Backtest Summary ----")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")
    if out_dir is not None:
        print(f"  reports: {out_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    setup_logger(config)

    try:
        if args.command in ("paper", "live"):
            asyncio.run(_run_live_or_paper(config, args.command))
        elif args.command == "backtest":
            asyncio.run(
                _run_backtest(config, args.start, args.end, args.config)
            )
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
