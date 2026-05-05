"""src/utils/path_utils.py 단위 테스트 (BL-1 Step A)."""

from __future__ import annotations

from pathlib import Path

from src.utils.path_utils import next_model_version, resolve_unique_dir


# ─── next_model_version ───


class TestNextModelVersion:
    def test_empty_root_returns_v001(self, tmp_path):
        models_root = tmp_path / "models" / "lightgbm"
        models_root.mkdir(parents=True)
        assert next_model_version(models_root) == "v001"

    def test_root_does_not_exist_returns_v001(self, tmp_path):
        models_root = tmp_path / "models" / "lightgbm"
        # mkdir 안 함
        assert next_model_version(models_root) == "v001"

    def test_basic_increment(self, tmp_path):
        """v001~v003 → 다음 v004."""
        models_root = tmp_path / "models" / "lightgbm"
        models_root.mkdir(parents=True)
        for n in (1, 2, 3):
            (models_root / f"v{n:03d}_15m_2020-01-01_2024-12-31").mkdir()
        assert next_model_version(models_root) == "v004"

    def test_skips_missing_v005(self, tmp_path):
        """v001~v004 + v006/v007 → max=7 → 다음 v008 (v005 비어있어도 자동 skip)."""
        models_root = tmp_path / "models" / "lightgbm"
        models_root.mkdir(parents=True)
        for n in (1, 2, 3, 4, 6, 7):
            (models_root / f"v{n:03d}_15m_2020-01-01_2024-12-31").mkdir()
        assert next_model_version(models_root) == "v008"

    def test_ignores_non_v_directories(self, tmp_path):
        models_root = tmp_path / "models" / "xgboost"
        models_root.mkdir(parents=True)
        (models_root / "v001_15m_2020-01-01_2024-12-31").mkdir()
        (models_root / "latest.json").touch()  # 파일 무시
        (models_root / "scratch").mkdir()  # v로 시작 안 함 무시
        assert next_model_version(models_root) == "v002"

    def test_handles_v005_only_existence(self, tmp_path):
        """v005만 단독 존재 → 다음 v006."""
        models_root = tmp_path / "models" / "lightgbm"
        models_root.mkdir(parents=True)
        (models_root / "v005_15m_2020-01-01_2024-12-31").mkdir()
        assert next_model_version(models_root) == "v006"

    def test_custom_prefix(self, tmp_path):
        models_root = tmp_path / "models" / "custom"
        models_root.mkdir(parents=True)
        (models_root / "m001_x").mkdir()
        (models_root / "m003_y").mkdir()
        assert next_model_version(models_root, prefix="m") == "m004"


# ─── resolve_unique_dir ───


class TestResolveUniqueDir:
    def test_nonexistent_returns_as_is(self, tmp_path):
        target = tmp_path / "report_260505_backtest"
        result = resolve_unique_dir(target)
        assert result == target

    def test_existing_returns_postfix_1(self, tmp_path):
        target = tmp_path / "report_260505_backtest"
        target.mkdir()
        result = resolve_unique_dir(target)
        assert result == tmp_path / "report_260505_backtest_1"

    def test_multiple_existing(self, tmp_path):
        target = tmp_path / "report_260505_backtest"
        target.mkdir()
        (tmp_path / "report_260505_backtest_1").mkdir()
        (tmp_path / "report_260505_backtest_2").mkdir()
        result = resolve_unique_dir(target)
        assert result == tmp_path / "report_260505_backtest_3"

    def test_no_filesystem_side_effect(self, tmp_path):
        """resolve만 함 — mkdir 부수효과 없음."""
        target = tmp_path / "newdir"
        result = resolve_unique_dir(target)
        assert not result.exists()  # mkdir 안 함

    def test_filling_postfix_gap(self, tmp_path):
        """_1, _3 존재 시 → _2가 비어있어도 순차 검사하므로 _2 반환."""
        target = tmp_path / "report"
        target.mkdir()
        (tmp_path / "report_1").mkdir()
        # _2 없음
        (tmp_path / "report_3").mkdir()
        result = resolve_unique_dir(target)
        assert result == tmp_path / "report_2"
