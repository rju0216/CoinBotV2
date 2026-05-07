"""src/utils/notifier.py 단위 테스트 (BL-2-1 Step 1)."""

from __future__ import annotations

import logging

import pytest

from src.utils.notifier import (
    CompositeNotifier,
    EmailNotifier,
    LogNotifier,
    Notifier,
    TelegramNotifier,
    build_notifier_from_config,
)


class TestLogNotifier:
    @pytest.mark.asyncio
    async def test_logs_at_correct_level(self, caplog):
        notifier = LogNotifier()
        with caplog.at_level(logging.WARNING):
            await notifier.send("WARNING", "Test", "Test message")
        assert any("[Test] Test message" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_includes_meta(self, caplog):
        notifier = LogNotifier()
        with caplog.at_level(logging.INFO):
            await notifier.send("INFO", "T", "M", strategy="ml_xgboost")
        assert any("strategy" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unknown_level_defaults_to_info(self, caplog):
        notifier = LogNotifier()
        with caplog.at_level(logging.INFO):
            await notifier.send("UNKNOWN", "T", "M")
        assert any(r.levelno == logging.INFO for r in caplog.records)


class TestTelegramNotifierFallback:
    @pytest.mark.asyncio
    async def test_empty_token_falls_back_to_log(self, caplog):
        notifier = TelegramNotifier(bot_token="", chat_id="123")
        with caplog.at_level(logging.WARNING):
            await notifier.send("INFO", "T", "M")
        # fallback warning + 실 메시지 둘 다 log에 기록
        assert any("미설정" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_chat_id_falls_back(self, caplog):
        notifier = TelegramNotifier(bot_token="abc", chat_id="")
        with caplog.at_level(logging.WARNING):
            await notifier.send("INFO", "T", "M")
        assert any("미설정" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_fallback_warned_only_once(self, caplog):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        with caplog.at_level(logging.WARNING):
            await notifier.send("INFO", "T1", "M1")
            await notifier.send("INFO", "T2", "M2")
            await notifier.send("INFO", "T3", "M3")
        warnings = [r for r in caplog.records if "미설정" in r.message]
        assert len(warnings) == 1


class TestTelegramTextFormat:
    """I-BL014: parse_mode 제거 + plain text 검증."""

    @pytest.mark.asyncio
    async def test_plain_text_no_markdown_chars(self, monkeypatch):
        """EXIT 메시지(특수 문자 포함)가 plain text로 송신됨 검증."""
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        captured = {}

        def mock_send_sync(text):
            captured["text"] = text

        monkeypatch.setattr(notifier, "_send_sync", mock_send_sync)
        await notifier.send(
            "INFO", "EXIT [ensemble] sl_hit", "net_pnl=$-40.71",
            strategy="ensemble", pnl=-40.71368847497047, reason="sl_hit",
        )
        text = captured["text"]
        # Markdown 데코레이션 사라짐
        assert "*[INFO]" not in text
        assert "_meta_" not in text
        assert "`{" not in text
        # 메시지 내용 plain text로 보존
        assert text.startswith("[INFO] EXIT [ensemble] sl_hit\n")
        assert "net_pnl=$-40.71" in text
        # meta plain key=value
        assert "meta: strategy=ensemble" in text
        assert "pnl=-40.71368847497047" in text
        assert "reason=sl_hit" in text

    @pytest.mark.asyncio
    async def test_no_meta_text_format(self, monkeypatch):
        """meta 없을 때 깔끔한 plain text."""
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        captured = {}
        monkeypatch.setattr(
            notifier, "_send_sync", lambda t: captured.setdefault("text", t)
        )
        await notifier.send("INFO", "Title", "Body content")
        assert captured["text"] == "[INFO] Title\nBody content"


class TestEmailNotifierFallback:
    @pytest.mark.asyncio
    async def test_empty_smtp_host_falls_back(self, caplog):
        notifier = EmailNotifier(
            smtp_host="", smtp_port=587, username="u", password="p",
            from_addr="from@x.com", to_addr="to@x.com",
        )
        with caplog.at_level(logging.WARNING):
            await notifier.send("INFO", "T", "M")
        assert any("미설정" in r.message for r in caplog.records)


class TestCompositeNotifier:
    @pytest.mark.asyncio
    async def test_calls_all_channels(self):
        calls = []

        class _Mock(Notifier):
            def __init__(self, name):
                self.name = name
            async def send(self, level, title, message, **meta):
                calls.append((self.name, level, title))

        comp = CompositeNotifier([_Mock("a"), _Mock("b"), _Mock("c")])
        await comp.send("INFO", "T", "M")
        assert sorted(c[0] for c in calls) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_one_channel_failure_doesnt_block_others(self):
        calls = []

        class _Failing(Notifier):
            async def send(self, level, title, message, **meta):
                raise RuntimeError("fail")

        class _Working(Notifier):
            async def send(self, level, title, message, **meta):
                calls.append("worked")

        comp = CompositeNotifier([_Failing(), _Working()])
        await comp.send("ERROR", "T", "M")
        assert calls == ["worked"]


class TestBuildFromConfig:
    def test_disabled_returns_log_only(self):
        cfg = {"live": {"notifications": {"enabled": False}}}
        n = build_notifier_from_config(cfg)
        assert isinstance(n, LogNotifier)

    def test_no_config_returns_log(self):
        n = build_notifier_from_config({})
        assert isinstance(n, LogNotifier)

    def test_log_only(self):
        cfg = {"live": {"notifications": {"enabled": True, "channels": ["log"]}}}
        n = build_notifier_from_config(cfg)
        assert isinstance(n, LogNotifier)

    def test_log_and_telegram_returns_composite(self):
        cfg = {
            "live": {
                "notifications": {
                    "enabled": True,
                    "channels": ["log", "telegram"],
                    "telegram": {"bot_token": "abc", "chat_id": "123"},
                }
            }
        }
        n = build_notifier_from_config(cfg)
        assert isinstance(n, CompositeNotifier)
        assert len(n.notifiers) == 2

    def test_empty_channels_returns_log(self):
        cfg = {"live": {"notifications": {"enabled": True, "channels": []}}}
        n = build_notifier_from_config(cfg)
        assert isinstance(n, LogNotifier)
