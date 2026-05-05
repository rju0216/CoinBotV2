"""알림 인프라 (BL-2-1, 사안 T''=가 log+telegram).

abstract Notifier + 구현체 4종 (Log / Telegram / Email / Composite).
config 기반 factory로 자동 인스턴스화. CoreEngine이 EventBus subscribe해서 호출.

설계 원칙:
- 기존 EventBus 패턴 재활용 (CoreEngine subscribe → notifier.send)
- notifier 자체는 단일 책임 (메시지 송신만). 이벤트 라우팅은 CoreEngine
- 텔레그램/이메일 미준비 시 자동 fallback (LogNotifier만 동작)
- 비동기 — 송신 지연이 거래 흐름 차단하지 않음

config 예시 (live.notifications):
  enabled: true
  channels: ["log"]                  # ["log", "telegram", "email"] 조합
  telegram:
    bot_token: ""                    # https://t.me/BotFather에서 발급
    chat_id: ""                      # 본인 chat ID
  email:
    smtp_host: ""
    smtp_port: 587
    username: ""
    password: ""
    from_addr: ""
    to_addr: ""
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Notifier(ABC):
    """abstract Notifier — 단일 책임 (메시지 송신)."""

    @abstractmethod
    async def send(self, level: str, title: str, message: str, **meta: Any) -> None:
        """level: 'INFO' / 'WARNING' / 'ERROR'. title: 짧은 헤더. message: 본문."""


class LogNotifier(Notifier):
    """logger 출력만. 항상 활성 (default)."""

    LEVEL_MAP = {
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    async def send(self, level: str, title: str, message: str, **meta: Any) -> None:
        log_level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        full_msg = f"[{title}] {message}"
        if meta:
            full_msg += f" | meta={meta}"
        logger.log(log_level, full_msg)


class TelegramNotifier(Notifier):
    """Telegram Bot API로 메시지 송신.

    bot_token, chat_id 미설정 시 LogNotifier로 자동 fallback (warning 1회).
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._fallback_warned = False

    async def send(self, level: str, title: str, message: str, **meta: Any) -> None:
        if not self.bot_token or not self.chat_id:
            if not self._fallback_warned:
                logger.warning(
                    "TelegramNotifier: bot_token/chat_id 미설정 — log fallback"
                )
                self._fallback_warned = True
            await LogNotifier().send(level, title, message, **meta)
            return
        # 비동기 HTTP — aiohttp 의존성 추가 부담. 단순화: requests 동기 호출을 to_thread로
        text = f"*[{level}] {title}*\n{message}"
        if meta:
            text += f"\n\n_meta_: `{meta}`"
        try:
            await asyncio.to_thread(self._send_sync, text)
        except Exception as e:
            logger.error("TelegramNotifier send failed: %s", e)

    def _send_sync(self, text: str) -> None:
        import urllib.parse
        import urllib.request
        url = self.API_URL.format(token=self.bot_token)
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Telegram API HTTP {resp.status}")


class EmailNotifier(Notifier):
    """SMTP 이메일 송신.

    smtp_host 등 미설정 시 LogNotifier fallback.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addr: str,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addr = to_addr
        self._fallback_warned = False

    async def send(self, level: str, title: str, message: str, **meta: Any) -> None:
        if not self.smtp_host or not self.from_addr or not self.to_addr:
            if not self._fallback_warned:
                logger.warning("EmailNotifier: SMTP 미설정 — log fallback")
                self._fallback_warned = True
            await LogNotifier().send(level, title, message, **meta)
            return
        try:
            await asyncio.to_thread(self._send_sync, level, title, message, meta)
        except Exception as e:
            logger.error("EmailNotifier send failed: %s", e)

    def _send_sync(self, level: str, title: str, message: str, meta: dict) -> None:
        import smtplib
        from email.mime.text import MIMEText
        body = f"[{level}] {title}\n\n{message}"
        if meta:
            body += f"\n\nmeta: {meta}"
        msg = MIMEText(body)
        msg["Subject"] = f"[CoinBot] [{level}] {title}"
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)


class CompositeNotifier(Notifier):
    """여러 채널에 동시 송신. 한 채널 실패해도 다른 채널 계속."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    async def send(self, level: str, title: str, message: str, **meta: Any) -> None:
        # 모든 채널 동시 송신 (gather로 병렬)
        await asyncio.gather(
            *(n.send(level, title, message, **meta) for n in self.notifiers),
            return_exceptions=True,
        )


def build_notifier_from_config(config: dict[str, Any]) -> Notifier:
    """config['live']['notifications']에서 channels 리스트로 Composite 자동 구성.

    채널 미설정 또는 enabled=false면 LogNotifier만 반환 (silent 안전).
    """
    notif_cfg = (config.get("live", {}) or {}).get("notifications", {}) or {}
    if not notif_cfg.get("enabled", False):
        return LogNotifier()
    channels = notif_cfg.get("channels", ["log"])
    notifiers: list[Notifier] = []
    if "log" in channels:
        notifiers.append(LogNotifier())
    if "telegram" in channels:
        tg = notif_cfg.get("telegram", {}) or {}
        notifiers.append(TelegramNotifier(
            bot_token=str(tg.get("bot_token", "")),
            chat_id=str(tg.get("chat_id", "")),
        ))
    if "email" in channels:
        em = notif_cfg.get("email", {}) or {}
        notifiers.append(EmailNotifier(
            smtp_host=str(em.get("smtp_host", "")),
            smtp_port=int(em.get("smtp_port", 587)),
            username=str(em.get("username", "")),
            password=str(em.get("password", "")),
            from_addr=str(em.get("from_addr", "")),
            to_addr=str(em.get("to_addr", "")),
        ))
    if not notifiers:
        return LogNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return CompositeNotifier(notifiers)
