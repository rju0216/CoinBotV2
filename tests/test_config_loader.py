"""config_loader 및 main CLI 파서 단위 테스트."""

from __future__ import annotations

import pytest

from src.utils.config_loader import load_config


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """테스트 격리: .env 파일 자동 로드 차단."""
    monkeypatch.setattr("src.utils.config_loader.load_dotenv", lambda: None)


def test_load_config_returns_parsed_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n"
        "  symbol: BTC/USDT:USDT\n"
        "  leverage: 5\n"
        "risk:\n"
        "  max_daily_loss_pct: 0.05\n",
        encoding="utf-8",
    )
    # env 완전 제거 → api_key 등이 주입되지 않음
    monkeypatch.delenv("OKX_API_KEY", raising=False)
    monkeypatch.delenv("OKX_SECRET", raising=False)
    monkeypatch.delenv("OKX_PASSPHRASE", raising=False)

    cfg = load_config(cfg_path)
    assert cfg["exchange"]["symbol"] == "BTC/USDT:USDT"
    assert cfg["exchange"]["leverage"] == 5
    assert cfg["risk"]["max_daily_loss_pct"] == 0.05
    assert "api_key" not in cfg["exchange"]


def test_load_config_injects_env_credentials(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n"
        "  symbol: BTC/USDT:USDT\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OKX_API_KEY", "test_key")
    monkeypatch.setenv("OKX_SECRET", "test_secret")
    monkeypatch.setenv("OKX_PASSPHRASE", "test_pw")

    cfg = load_config(cfg_path)
    assert cfg["exchange"]["api_key"] == "test_key"
    assert cfg["exchange"]["secret"] == "test_secret"
    assert cfg["exchange"]["passphrase"] == "test_pw"


def test_load_config_injects_telegram_credentials(tmp_path, monkeypatch):
    """BL-2-1: TELEGRAM_BOT_TOKEN/CHAT_ID env → live.notifications.telegram에 주입."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n  symbol: BTC/USDT:USDT\n"
        "live:\n  notifications:\n    enabled: true\n    channels: [\"log\", \"telegram\"]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test_chat")

    cfg = load_config(cfg_path)
    tg = cfg["live"]["notifications"]["telegram"]
    assert tg["bot_token"] == "test_token"
    assert tg["chat_id"] == "test_chat"


def test_load_config_creates_telegram_section_if_absent(tmp_path, monkeypatch):
    """BL-2-1: notifications.telegram 섹션이 없어도 env가 있으면 자동 생성."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("exchange:\n  symbol: BTC/USDT:USDT\n", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    cfg = load_config(cfg_path)
    assert cfg["live"]["notifications"]["telegram"]["bot_token"] == "abc"
    assert cfg["live"]["notifications"]["telegram"]["chat_id"] == "123"


def test_load_config_no_telegram_env_keeps_yaml_default(tmp_path, monkeypatch):
    """env 미설정 시 yaml의 기본값 (빈 문자열) 그대로."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n  symbol: BTC/USDT:USDT\n"
        "live:\n  notifications:\n    telegram:\n      bot_token: \"\"\n      chat_id: \"\"\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    cfg = load_config(cfg_path)
    assert cfg["live"]["notifications"]["telegram"]["bot_token"] == ""


def test_load_config_injects_email_credentials(tmp_path, monkeypatch):
    """BL-2-1: EMAIL_SMTP_USERNAME/PASSWORD env → live.notifications.email에 주입."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n  symbol: BTC/USDT:USDT\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMAIL_SMTP_USERNAME", "user@x.com")
    monkeypatch.setenv("EMAIL_SMTP_PASSWORD", "pw")

    cfg = load_config(cfg_path)
    em = cfg["live"]["notifications"]["email"]
    assert em["username"] == "user@x.com"
    assert em["password"] == "pw"


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_creates_exchange_section_if_absent(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("risk:\n  max_daily_loss_pct: 0.05\n", encoding="utf-8")
    monkeypatch.setenv("OKX_API_KEY", "x")
    cfg = load_config(cfg_path)
    assert cfg["exchange"]["api_key"] == "x"


# ---- CLI 파서 ----


def test_cli_requires_subcommand():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args([])  # subcommand 미지정


def test_cli_paper_requires_config():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["paper"])  # --config 누락


def test_cli_backtest_requires_start_end():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["backtest", "--config", "x.yaml"])


def test_cli_backtest_valid():
    from src.main import _parse_args
    args = _parse_args([
        "backtest", "--config", "x.yaml",
        "--start", "2024-01-01", "--end", "2024-12-31",
    ])
    assert args.command == "backtest"
    assert args.config == "x.yaml"
    assert args.start == "2024-01-01"
    assert args.end == "2024-12-31"


def test_cli_paper_valid():
    from src.main import _parse_args
    args = _parse_args(["paper", "--config", "x.yaml"])
    assert args.command == "paper"
    assert args.config == "x.yaml"


def test_cli_live_valid():
    from src.main import _parse_args
    args = _parse_args(["live", "--config", "x.yaml"])
    assert args.command == "live"


def test_cli_unknown_subcommand_rejected():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["unknown", "--config", "x.yaml"])


def test_default_yaml_has_required_keys(monkeypatch):
    """default.yaml이 단계 2~10에서 신설된 config 키를 모두 포함하는지."""
    monkeypatch.delenv("OKX_API_KEY", raising=False)
    monkeypatch.delenv("OKX_SECRET", raising=False)
    monkeypatch.delenv("OKX_PASSPHRASE", raising=False)
    cfg = load_config("config/default.yaml")
    # I-003: paper.initial_balance
    assert "paper" in cfg
    assert "initial_balance" in cfg["paper"]
    # I-004: exchange.leverage
    assert "leverage" in cfg["exchange"]
    # 기타 필수 섹션
    assert cfg["engine"]["reverse_signal_policy"] in (
        "ignore", "reverse", "same_strategy_only"
    )
    assert "max_concurrent_positions" in cfg["risk"]
    assert "taker_fee_pct" in cfg["accounting"]
    assert "active" in cfg["strategies"]
    # 활성 전략은 필수 키 보유
    for name in cfg["strategies"]["active"]:
        assert "risk_per_trade_pct" in cfg[name]
        assert "max_leverage" in cfg[name]
