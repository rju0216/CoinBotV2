"""모델 통합 평가 스크립트 (Phase E-2).

5개 모델 × 6개 OOS 분할 자동 백테 + 베이스라인 비교 + 결과 수집.

분할 매트릭스:
  - 분할 1 (v001, 5년 학습): OOS 2025-01-01 ~ 2025-12-31  (1년)
  - Anchored A (v003, 3년): OOS 2023-01-01 ~ 2023-12-31    (1년)
  - Anchored B (v004, 4년): OOS 2024-01-01 ~ 2024-12-31    (1년)
  - Expanding 2 (v004, 4년): OOS 2024-01-01 ~ 2025-12-31   (2년)
  - Expanding 3 (v003, 3년): OOS 2023-01-01 ~ 2025-12-31   (3년)
  - Expanding 4 (v002, 2년): OOS 2022-01-01 ~ 2025-12-31   (4년)

출력: data/backtest_reports/00_Working/eval_{YYMMDD}/
  - {strategy}_{split}/   각 백테 결과 (trades.csv, equity_curve.csv, metrics.json, ...)
  - comparison.csv        종합 비교 표

Usage:
    # E-2-2 (전체 자동 실행):
    python scripts/evaluate_models.py --mode full

    # 특정 모델만:
    python scripts/evaluate_models.py --mode single --strategy ml_lightgbm --split A

    # 결과 수집만 (이미 백테 끝났을 때):
    python scripts/evaluate_models.py --mode collect

E-2-3/E-2-4에서 슬리피지·calibration·통계 분석 추가 예정.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_BASE = Path("data/backtest_reports/00_Working")


@dataclass
class BacktestSpec:
    """단일 백테 명세."""
    strategy: str            # "ml_lightgbm", "ml_xgboost", ...
    config_path: str         # config YAML 경로
    model_dir: str           # 모델 디렉토리 (model_path 오버라이드용)
    split_id: str            # "1", "A", "B", "Exp2", "Exp3", "Exp4"
    oos_start: str           # "2025-01-01"
    oos_end: str             # "2025-12-31"

    @property
    def label(self) -> str:
        """결과 디렉토리 이름."""
        return f"{self.strategy}_{self.split_id}"


# ─── 모델 × 분할 매트릭스 ───

STRATEGIES = {
    "ml_lightgbm":   "config/ml_lightgbm.yaml",
    "ml_xgboost":    "config/ml_xgboost.yaml",
    "dl_lstm":       "config/dl_lstm.yaml",
    "dl_transformer": "config/dl_transformer.yaml",
    "rl_ppo":        "config/rl_ppo.yaml",
}

# 모델 디렉토리 폴더명 패턴: v{NNN}_15m_2020-01-01_{end}
MODEL_VERSIONS = {
    "v001": "v001_15m_2020-01-01_2024-12-31",  # 5년
    "v002": "v002_15m_2020-01-01_2021-12-31",  # 2년
    "v003": "v003_15m_2020-01-01_2022-12-31",  # 3년
    "v004": "v004_15m_2020-01-01_2023-12-31",  # 4년
}

# 모델 종류 → 디렉토리 prefix (xgboost는 ml_xgboost가 아닌 xgboost)
STRATEGY_DIR_PREFIX = {
    "ml_lightgbm": "lightgbm",
    "ml_xgboost":  "xgboost",
    "dl_lstm":     "lstm",
    "dl_transformer": "transformer",
    "rl_ppo":      "ppo",
}

# (split_id, model_version, oos_start, oos_end)
SPLIT_DEFINITIONS = [
    ("1",    "v001", "2025-01-01", "2025-12-31"),  # Anchored C / Expanding 1
    ("A",    "v003", "2023-01-01", "2023-12-31"),  # Anchored A
    ("B",    "v004", "2024-01-01", "2024-12-31"),  # Anchored B
    ("Exp2", "v004", "2024-01-01", "2025-12-31"),  # Expanding 2 (2년 OOS)
    ("Exp3", "v003", "2023-01-01", "2025-12-31"),  # Expanding 3 (3년 OOS)
    ("Exp4", "v002", "2022-01-01", "2025-12-31"),  # Expanding 4 (4년 OOS)
]


def build_specs(strategies: list[str] | None = None) -> list[BacktestSpec]:
    """모델 × 분할 매트릭스 → BacktestSpec 리스트.

    strategies가 None이면 전체 5개. 일부만 받으면 그 모델만.
    """
    target_strategies = strategies or list(STRATEGIES.keys())
    specs: list[BacktestSpec] = []
    for strat in target_strategies:
        if strat not in STRATEGIES:
            logger.warning("알 수 없는 strategy: %s — 스킵", strat)
            continue
        config_path = STRATEGIES[strat]
        dir_prefix = STRATEGY_DIR_PREFIX[strat]
        for split_id, version, oos_start, oos_end in SPLIT_DEFINITIONS:
            model_dir = f"models/{dir_prefix}/{MODEL_VERSIONS[version]}"
            specs.append(BacktestSpec(
                strategy=strat,
                config_path=config_path,
                model_dir=model_dir,
                split_id=split_id,
                oos_start=oos_start,
                oos_end=oos_end,
            ))
    return specs


def _override_model_path(config: dict[str, Any], strategy: str, model_dir: str) -> dict[str, Any]:
    """config의 strategy 섹션에서 model_path를 명시적 디렉토리로 오버라이드."""
    if strategy in config and isinstance(config[strategy], dict):
        config[strategy]["model_path"] = model_dir
    return config


async def run_one_backtest(spec: BacktestSpec, eval_root: Path) -> tuple[Path, dict] | None:
    """단일 백테 실행 → (결과 디렉토리, metrics dict) 반환.

    실패 시 None.
    """
    logger.info("[BACKTEST] %s | %s | OOS %s ~ %s",
                spec.strategy, spec.split_id, spec.oos_start, spec.oos_end)

    config = load_config(spec.config_path)
    config = _override_model_path(config, spec.strategy, spec.model_dir)

    engine = BacktestEngine(config, start=spec.oos_start, end=spec.oos_end)
    out_dir = None
    try:
        await engine.initialize()
        await engine.run()
        result = await engine.get_result()
        # eval_root 하위에 spec.label 디렉토리로 저장
        out_dir = engine.write_reports(
            config_path=spec.config_path,
            out_root=eval_root / spec.label,
        )
    except Exception:
        logger.exception("백테 실패: %s", spec.label)
        return None
    finally:
        await engine.shutdown()

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error("metrics.json 없음: %s", metrics_path)
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    logger.info("[OK] %s → %s (trades=%d, return=%.2f%%)",
                spec.label, out_dir,
                metrics["integrated"]["total_trades"],
                metrics["integrated"]["total_return_pct"])
    return out_dir, metrics


def collect_metrics(eval_root: Path, specs: list[BacktestSpec]) -> pd.DataFrame:
    """eval_root 하위의 모든 metrics.json 수집 → 비교 DataFrame."""
    rows: list[dict] = []
    for spec in specs:
        # write_reports 내부 구조: {eval_root/spec.label}/{config_name}/metrics.json
        config_name = Path(spec.config_path).stem
        metrics_path = eval_root / spec.label / config_name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("metrics 없음: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        integ = m.get("integrated", {})
        rows.append({
            "strategy": spec.strategy,
            "split": spec.split_id,
            "oos_start": spec.oos_start,
            "oos_end": spec.oos_end,
            "total_trades": integ.get("total_trades"),
            "win_rate_pct": integ.get("win_rate_pct"),
            "total_return_pct": integ.get("total_return_pct"),
            "max_drawdown_pct": integ.get("max_drawdown_pct"),
            "profit_factor": integ.get("profit_factor"),
            "avg_win": integ.get("avg_win"),
            "avg_loss": integ.get("avg_loss"),
        })
    return pd.DataFrame(rows)


async def run_all(specs: list[BacktestSpec], eval_root: Path) -> None:
    """specs 전체를 순차 실행."""
    eval_root.mkdir(parents=True, exist_ok=True)
    logger.info("총 %d개 백테 시작 (출력: %s)", len(specs), eval_root)
    n_success = 0
    for i, spec in enumerate(specs, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i, len(specs), spec.label)
        result = await run_one_backtest(spec, eval_root)
        if result is not None:
            n_success += 1
    logger.info("=" * 60)
    logger.info("백테 완료: 성공 %d / 전체 %d", n_success, len(specs))


def save_comparison(df: pd.DataFrame, eval_root: Path) -> Path:
    """비교 표 CSV 저장."""
    eval_root.mkdir(parents=True, exist_ok=True)
    csv_path = eval_root / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("비교 표 저장: %s (%d rows)", csv_path, len(df))
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="모델 통합 평가 (Phase E-2)")
    parser.add_argument(
        "--mode",
        choices=["full", "single", "collect"],
        default="full",
        help="full: 모든 모델×분할 백테 / single: 특정 조합 / collect: 결과 수집만",
    )
    parser.add_argument("--strategy", help="--mode single 시 특정 strategy 이름")
    parser.add_argument("--split", help="--mode single 시 특정 split id")
    parser.add_argument(
        "--eval-date",
        default=datetime.now().strftime("%y%m%d"),
        help="결과 디렉토리 날짜 prefix (기본: 오늘)",
    )
    args = parser.parse_args()

    eval_root = REPORT_BASE / f"eval_{args.eval_date}"

    if args.mode == "full":
        specs = build_specs()
        asyncio.run(run_all(specs, eval_root))
        df = collect_metrics(eval_root, specs)
        save_comparison(df, eval_root)
    elif args.mode == "single":
        if not args.strategy or not args.split:
            logger.error("--mode single은 --strategy와 --split 필요")
            return 2
        all_specs = build_specs([args.strategy])
        specs = [s for s in all_specs if s.split_id == args.split]
        if not specs:
            logger.error("매칭되는 spec 없음: strategy=%s split=%s",
                         args.strategy, args.split)
            return 2
        asyncio.run(run_all(specs, eval_root))
    elif args.mode == "collect":
        specs = build_specs()
        df = collect_metrics(eval_root, specs)
        save_comparison(df, eval_root)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
