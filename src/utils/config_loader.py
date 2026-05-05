"""YAML config 로더 + .env 기반 자격증명/시크릿 주입.

config.yaml에는 시크릿(API key, bot token 등) 두지 말 것.
모두 .env (또는 OS 환경변수)로 주입 — git에서 .env 제외 (.gitignore).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """YAML config 파일을 로드하고 환경변수에서 자격증명을 주입한다.

    환경변수 (.env 또는 OS 환경):
      OKX_API_KEY / OKX_SECRET / OKX_PASSPHRASE  → exchange.{api_key, secret, passphrase}
      TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID      → live.notifications.telegram.{bot_token, chat_id}
      EMAIL_SMTP_USERNAME / EMAIL_SMTP_PASSWORD  → live.notifications.email.{username, password}
    """
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    # OKX 자격증명
    exchange = config.setdefault("exchange", {})
    if os.getenv("OKX_API_KEY") is not None:
        exchange["api_key"] = os.getenv("OKX_API_KEY", "")
    if os.getenv("OKX_SECRET") is not None:
        exchange["secret"] = os.getenv("OKX_SECRET", "")
    if os.getenv("OKX_PASSPHRASE") is not None:
        exchange["passphrase"] = os.getenv("OKX_PASSPHRASE", "")

    # BL-2-1: Telegram 봇 자격증명 — config의 live.notifications.telegram.bot_token/chat_id에 주입.
    # config에 빈 문자열로 두고 .env에서 실값 주입 (보안)
    if os.getenv("TELEGRAM_BOT_TOKEN") is not None or os.getenv("TELEGRAM_CHAT_ID") is not None:
        live = config.setdefault("live", {})
        notif = live.setdefault("notifications", {})
        tg = notif.setdefault("telegram", {})
        if os.getenv("TELEGRAM_BOT_TOKEN") is not None:
            tg["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if os.getenv("TELEGRAM_CHAT_ID") is not None:
            tg["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")

    # Email SMTP 자격증명 (옵션)
    if os.getenv("EMAIL_SMTP_USERNAME") is not None or os.getenv("EMAIL_SMTP_PASSWORD") is not None:
        live = config.setdefault("live", {})
        notif = live.setdefault("notifications", {})
        em = notif.setdefault("email", {})
        if os.getenv("EMAIL_SMTP_USERNAME") is not None:
            em["username"] = os.getenv("EMAIL_SMTP_USERNAME", "")
        if os.getenv("EMAIL_SMTP_PASSWORD") is not None:
            em["password"] = os.getenv("EMAIL_SMTP_PASSWORD", "")

    return config
