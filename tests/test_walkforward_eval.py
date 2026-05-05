"""evaluate_models walkforward 모드 단위 테스트 (BL-1-3).

mock fold 디렉토리 + train_meta.json + 가짜 백테 결과로 build_walkforward_specs +
aggregate_walkforward_results 검증.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.evaluate_models import (
    BacktestSpec,
    aggregate_walkforward_results,
    build_walkforward_specs,
)


def _make_fold_meta(
    fold_dir: Path,
    fold_id: int,
    test_start: str,
    test_end: str,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "fold_id": fold_id,
        "train_start": "2020-01-01T00:00:00+00:00",
        "train_end": "2020-06-30T00:00:00+00:00",
        "test_start": f"{test_start}T00:00:00+00:00",
        "test_end": f"{test_end}T00:00:00+00:00",
        "oos_accuracy": 0.7,
        "oos_f1_macro": 0.7,
        "label_params": {"method": "triple_barrier"},
    }
    (fold_dir / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _make_fold_backtest_result(
    eval_root: Path,
    strategy: str,
    fold_id: int,
    config_name: str,
    *,
    total_pnl: float,
    total_trades: int,
    winning: int,
    losing: int,
    return_pct: float,
    mdd: float,
    gross_profit: float,
    gross_loss: float,
    calibration_method: str | None = "none",
) -> None:
    """spec.label = f'{strategy}_fold_NN_calnone' → eval_root/{label}/{config_name}/metrics.json.

    BacktestSpec.label가 calibration suffix를 포함하므로 동일 패턴 유지.
    """
    label = f"{strategy}_fold_{fold_id:02d}"
    if calibration_method is not None:
        label += f"_cal{calibration_method}"
    out = eval_root / label / config_name
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "integrated": {
            "initial_balance": 10000.0,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "total_return_pct": return_pct,
            "max_drawdown_pct": mdd,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
    }
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    # 빈 trades.csv
    pd.DataFrame(
        [{"id": 1, "pnl": total_pnl, "side": "long"}]
    ).to_csv(out / "trades.csv", index=False)


class TestBuildWalkforwardSpecs:
    def test_scans_fold_dirs(self, tmp_path, monkeypatch):
        # mock 모델 디렉토리 구조: tmp_path/models/lightgbm/v001/folds/fold_00, fold_01
        models_dir = tmp_path / "models" / "lightgbm" / "v001_15m_2020_2024"
        folds_dir = models_dir / "folds"
        _make_fold_meta(folds_dir / "fold_00", 0, "2020-07-01", "2020-08-31")
        _make_fold_meta(folds_dir / "fold_01", 1, "2020-09-01", "2020-10-31")
        # latest.json
        latest_path = tmp_path / "models" / "lightgbm" / "latest.json"
        latest_path.write_text(json.dumps({"path": str(models_dir)}), encoding="utf-8")

        # _resolve_strategy_model_dir이 Path("models")부터 시작 → cwd 변경 필요
        monkeypatch.chdir(tmp_path)
        specs = build_walkforward_specs("ml_lightgbm")

        assert len(specs) == 2
        assert all(isinstance(s, BacktestSpec) for s in specs)
        assert specs[0].split_id == "fold_00"
        assert specs[0].oos_start == "2020-07-01"
        assert specs[0].oos_end == "2020-08-31"
        assert specs[0].calibration_method == "none"  # 사안 L'
        assert specs[0].model_dir == str(folds_dir / "fold_00")
        assert specs[1].split_id == "fold_01"

    def test_missing_latest_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="latest.json"):
            build_walkforward_specs("ml_lightgbm")

    def test_missing_folds_dir_raises(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models" / "lightgbm" / "v001_15m_2020_2024"
        models_dir.mkdir(parents=True, exist_ok=True)
        latest_path = tmp_path / "models" / "lightgbm" / "latest.json"
        latest_path.write_text(json.dumps({"path": str(models_dir)}), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="folds/"):
            build_walkforward_specs("ml_lightgbm")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_walkforward_specs("nonexistent_model")


class TestAggregateWalkforwardResults:
    def test_basic_aggregation(self, tmp_path):
        eval_root = tmp_path / "eval_test"
        config_name = "ml_lightgbm"
        # 3 folds 가짜 결과 생성
        _make_fold_backtest_result(
            eval_root, "ml_lightgbm", 0, config_name,
            total_pnl=1000, total_trades=10, winning=7, losing=3,
            return_pct=10.0, mdd=2.5,
            gross_profit=2000, gross_loss=1000,
        )
        _make_fold_backtest_result(
            eval_root, "ml_lightgbm", 1, config_name,
            total_pnl=500, total_trades=8, winning=5, losing=3,
            return_pct=5.0, mdd=1.8,
            gross_profit=1500, gross_loss=1000,
        )
        _make_fold_backtest_result(
            eval_root, "ml_lightgbm", 2, config_name,
            total_pnl=-200, total_trades=12, winning=4, losing=8,
            return_pct=-2.0, mdd=4.0,
            gross_profit=800, gross_loss=1000,
        )

        specs = [
            BacktestSpec(
                strategy="ml_lightgbm",
                config_path=f"config/{config_name}.yaml",
                model_dir=f"/tmp/folds/fold_{i:02d}",
                split_id=f"fold_{i:02d}",
                oos_start="2020-07-01",
                oos_end="2020-08-31",
                calibration_method="none",
            )
            for i in range(3)
        ]

        out_dir = aggregate_walkforward_results(eval_root, "ml_lightgbm", specs)

        assert out_dir.exists()
        assert (out_dir / "walkforward_metrics.json").exists()
        assert (out_dir / "walkforward_trades.csv").exists()

        with open(out_dir / "walkforward_metrics.json") as f:
            agg = json.load(f)
        assert agg["n_folds"] == 3
        assert agg["total_trades"] == 30  # 10+8+12
        assert agg["winning_trades"] == 16  # 7+5+4
        assert agg["losing_trades"] == 14  # 3+3+8
        assert agg["pooled_win_rate_pct"] == pytest.approx(53.33, abs=0.01)
        assert agg["sum_pnl"] == 1300  # 1000+500-200
        # 사안 K' (가): mean_return_pct = (10 + 5 + (-2)) / 3 = 4.33
        assert agg["mean_return_pct_per_fold"] == pytest.approx(4.3333, abs=0.001)
        # max_drawdown = 4.0 (fold 2)
        assert agg["max_drawdown_pct_max"] == 4.0
        # PF = 4300 / 3000 = 1.4333
        assert agg["profit_factor"] == pytest.approx(1.4333, abs=0.001)

    def test_empty_metrics_raises(self, tmp_path):
        eval_root = tmp_path / "eval_empty"
        specs = [
            BacktestSpec(
                strategy="ml_lightgbm",
                config_path="config/ml_lightgbm.yaml",
                model_dir="/tmp/none",
                split_id="fold_00",
                oos_start="2020-07-01",
                oos_end="2020-08-31",
            )
        ]
        with pytest.raises(RuntimeError, match="통합 가능한 fold 결과 없음"):
            aggregate_walkforward_results(eval_root, "ml_lightgbm", specs)
