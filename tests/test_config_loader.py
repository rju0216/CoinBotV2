"""config_loader 및 main CLI 파서 단위 테스트."""

from __future__ import annotations

import pytest

from src.utils.config_loader import load_config


def test_load_config_returns_parsed_yaml(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "exchange:\n"
        "  symbol: BTC/USDT:USDT\n"
        "accounting:\n"
        "  taker_fee_pct: 0.0005\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg["exchange"]["symbol"] == "BTC/USDT:USDT"
    assert cfg["accounting"]["taker_fee_pct"] == 0.0005


def test_load_config_empty_file_returns_empty_dict(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("", encoding="utf-8")
    assert load_config(cfg_path) == {}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_default_yaml_has_required_keys():
    """default.yaml 이 백테 골격 필수 키를 모두 포함하는지."""
    cfg = load_config("config/default.yaml")
    assert cfg["exchange"]["symbol"]
    assert "taker_fee_pct" in cfg["accounting"]
    assert "initial_balance" in cfg["backtest"]
    assert "history_bars" in cfg["data"]
    assert "active" in cfg["strategies"]
    # 뼈대 상태 — active 는 비어 있음 (거래 정책은 전략 메서드 소유)
    assert cfg["strategies"]["active"] == []


# ---- CLI 파서 (backtest 전용) ----


def test_cli_requires_subcommand():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args([])


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


def test_cli_unknown_subcommand_rejected():
    from src.main import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["unknown", "--config", "x.yaml"])
