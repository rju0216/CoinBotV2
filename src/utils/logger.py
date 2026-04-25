"""로깅 초기화 — rich 콘솔 + 회전 파일 핸들러."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from rich.logging import RichHandler


def setup_logger(config: dict[str, Any]) -> logging.Logger:
    """config.logging 섹션을 기반으로 루트 로거를 초기화.

    파일명에 실행 시작 시각을 접미사로 부여하여 재시작 로그 분리.
    """
    log_cfg = config.get("logging", {}) or {}
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "logs/coinbot.log")
    max_bytes = int(log_cfg.get("max_size_mb", 50)) * 1024 * 1024
    backup_count = int(log_cfg.get("backup_count", 5))

    log_dir = os.path.dirname(log_file) or "."
    log_base = os.path.splitext(os.path.basename(log_file))[0]
    log_ext = os.path.splitext(log_file)[1] or ".log"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_base}_{timestamp}{log_ext}")
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console = RichHandler(
            rich_tracebacks=True, show_time=True, show_path=False
        )
        console.setLevel(level)
        root_logger.addHandler(console)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    return root_logger
