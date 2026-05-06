"""StrategyModule.extract_train_meta default impl 단위 테스트 (I-BL003 fix)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule


class _DummyPlugin(StrategyModule):
    name = "_dummy_plugin"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        raise NotImplementedError

    def compute_stop_loss(self, ctx, signal):
        raise NotImplementedError

    def compute_take_profit(self, ctx, signal, stop_loss):
        raise NotImplementedError


def _write_train_meta(model_dir, period: str, oos_acc: float | None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {"train_period": period}
    if oos_acc is not None:
        meta["oos_accuracy"] = oos_acc
    (model_dir / "train_meta.json").write_text(json.dumps(meta))


class TestExtractTrainMetaDefault:
    """단일 모델 default 동작 검증 — 기존 engine._extract_train_meta 회귀 보호."""

    def test_direct_model_path(self, tmp_path):
        """model_path가 직접 model_dir을 가리킬 때 train_meta.json 정상 추출."""
        model_dir = tmp_path / "lightgbm" / "v009_15m_2020-01-01_2024-12-31"
        _write_train_meta(model_dir, "2020-01-01 ~ 2024-12-31", 0.7521)

        plugin = _DummyPlugin({"model_path": str(model_dir)})
        cutoff, acc = plugin.extract_train_meta()

        assert cutoff == datetime(2024, 12, 31, tzinfo=timezone.utc)
        assert acc == pytest.approx(0.7521)

    def test_latest_json_redirect(self, tmp_path):
        """model_path가 'latest'로 끝나면 latest.json 경유 해석."""
        version_dir = tmp_path / "transformer" / "v009_15m_2020-01-01_2026-05-04"
        _write_train_meta(version_dir, "2020-01-01 ~ 2026-05-04", 0.7612)
        # latest.json은 model_dir.parent에 위치
        (tmp_path / "transformer" / "latest.json").write_text(
            json.dumps({"path": str(version_dir)})
        )

        plugin = _DummyPlugin({"model_path": str(tmp_path / "transformer" / "latest")})
        cutoff, acc = plugin.extract_train_meta()

        assert cutoff == datetime(2026, 5, 4, tzinfo=timezone.utc)
        assert acc == pytest.approx(0.7612)

    def test_no_model_path(self):
        """model_path 없으면 (None, None)."""
        plugin = _DummyPlugin({})
        assert plugin.extract_train_meta() == (None, None)

    def test_train_meta_missing(self, tmp_path):
        """model_dir 존재하지만 train_meta.json 없으면 (None, None)."""
        model_dir = tmp_path / "xgboost" / "v001"
        model_dir.mkdir(parents=True)

        plugin = _DummyPlugin({"model_path": str(model_dir)})
        assert plugin.extract_train_meta() == (None, None)

    def test_oos_accuracy_missing(self, tmp_path):
        """train_period만 있고 oos_accuracy 키 없을 때 cutoff만 반환, acc는 None."""
        model_dir = tmp_path / "lstm" / "v009"
        _write_train_meta(model_dir, "2020-01-01 ~ 2026-05-04", None)

        plugin = _DummyPlugin({"model_path": str(model_dir)})
        cutoff, acc = plugin.extract_train_meta()

        assert cutoff == datetime(2026, 5, 4, tzinfo=timezone.utc)
        assert acc is None

    def test_train_period_malformed(self, tmp_path):
        """train_period 파싱 불가 시 (None, None) — 안전 fallback."""
        model_dir = tmp_path / "lightgbm" / "v_broken"
        model_dir.mkdir(parents=True)
        (model_dir / "train_meta.json").write_text(
            json.dumps({"train_period": "garbage", "oos_accuracy": 0.7})
        )

        plugin = _DummyPlugin({"model_path": str(model_dir)})
        assert plugin.extract_train_meta() == (None, None)
