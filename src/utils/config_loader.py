"""YAML config 로더 + .env 기반 API 자격증명 주입."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """YAML config 파일을 로드하고 OKX API 자격증명을 환경변수에서 주입한다.

    환경변수 (.env 또는 OS 환경):
      OKX_API_KEY / OKX_SECRET / OKX_PASSPHRASE

    config.yaml의 exchange 섹션에는 이 키를 두지 말 것 (보안).
    """
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    exchange = config.setdefault("exchange", {})
    # 값이 있을 때만 주입 (없으면 load_config 직접 호출 테스트 편의)
    if os.getenv("OKX_API_KEY") is not None:
        exchange["api_key"] = os.getenv("OKX_API_KEY", "")
    if os.getenv("OKX_SECRET") is not None:
        exchange["secret"] = os.getenv("OKX_SECRET", "")
    if os.getenv("OKX_PASSPHRASE") is not None:
        exchange["passphrase"] = os.getenv("OKX_PASSPHRASE", "")

    return config
