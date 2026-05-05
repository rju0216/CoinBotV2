"""evaluate_models 결과 분석 스크립트 (Phase E-2-4 Step 2).

eval_<날짜>_baseline/ 폴더 자동 스캔 → 30 specs(5 모델 × 6 분할) 매트릭스
+ 5 모델 pairwise bootstrap (분할 1, Exp4 양 끝점) 출력.

출력:
  eval_<날짜>_baseline/analysis_metrics.csv  — 매트릭스 (Sharpe/Calmar/PF/MDD/Win rate)
  eval_<날짜>_baseline/bootstrap_pvalues.csv — 5 모델 pairwise (분할 1, Exp4)

베이스라인 (macross/B&H)는 metrics 매트릭스에 포함하되, B&H는 equity_curve가 없어
Sharpe 계산 불가 → 해당 셀만 NaN.

Usage:
    python scripts/analyze_results.py --eval-date 260503_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml.metrics_extended import (  # noqa: E402
    bonferroni_correction,
    bootstrap_pnl_diff,
    compute_calmar_ratio,
    compute_sharpe_ratio,
    fdr_correction,
    split_to_oos_years,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_BASE = Path("data/backtest_reports/00_Working")

# evaluate_models.py SPLIT_DEFINITIONS와 일관 (코드 중복 회피)
SPLIT_IDS = ["1", "A", "B", "Exp2", "Exp3", "Exp4"]
MODEL_STRATEGIES = ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer", "rl_ppo"]
BOOTSTRAP_TARGET_SPLITS = ["1", "Exp4"]  # 양 끝점 (사안 D)


def _spec_dir(eval_root: Path, strategy: str, split_id: str) -> Path:
    """{strategy}_{split_id}/{config_name}/ 경로 — config_name은 strategy와 동일 (evaluate_models 패턴)."""
    return eval_root / f"{strategy}_{split_id}" / strategy


def _macross_dir(eval_root: Path, split_id: str) -> Path:
    """macross_{split_id}/_eval_macross_{split_id}/ 경로."""
    return eval_root / f"macross_{split_id}" / f"_eval_macross_{split_id}"


def _load_metrics_row(cfg_dir: Path, strategy: str, split_id: str) -> dict | None:
    """단일 spec 폴더 → metrics + Sharpe + Calmar 한 row."""
    metrics_path = cfg_dir / "metrics.json"
    equity_path = cfg_dir / "equity_curve.csv"
    if not metrics_path.exists():
        logger.warning("metrics 없음: %s", metrics_path)
        return None

    with open(metrics_path) as f:
        m = json.load(f).get("integrated", {})

    sharpe = 0.0
    if equity_path.exists():
        eq = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        if eq.index.tz is None:
            eq.index = eq.index.tz_localize("UTC")
        sharpe = compute_sharpe_ratio(eq)

    oos_years = split_to_oos_years(split_id)
    total_return = float(m.get("total_return_pct", 0.0))
    max_dd = float(m.get("max_drawdown_pct", 0.0))
    calmar = compute_calmar_ratio(total_return, max_dd, oos_years)

    pf_raw = m.get("profit_factor")
    pf = float(pf_raw) if isinstance(pf_raw, (int, float)) else None

    return {
        "strategy": strategy,
        "split": split_id,
        "oos_years": oos_years,
        "total_trades": int(m.get("total_trades", 0)),
        "total_return_pct": round(total_return, 2),
        "win_rate_pct": float(m.get("win_rate_pct", 0.0)),
        "profit_factor": pf,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "calmar_ratio": round(calmar, 3),
    }


def collect_metrics_matrix(eval_root: Path) -> pd.DataFrame:
    """30 specs (5 모델 × 6 분할) + macross 6 → 통합 매트릭스. B&H는 별도 처리(Sharpe 불가)."""
    rows: list[dict] = []

    # 모델 30 specs
    for strat in MODEL_STRATEGIES:
        for split in SPLIT_IDS:
            row = _load_metrics_row(_spec_dir(eval_root, strat, split), strat, split)
            if row:
                rows.append(row)

    # macross 6
    for split in SPLIT_IDS:
        row = _load_metrics_row(_macross_dir(eval_root, split), "example_macross", split)
        if row:
            rows.append(row)

    # B&H 6 (buy_and_hold.json — Sharpe/Calmar 불가, total_return + MDD만)
    bh_path = eval_root / "buy_and_hold.json"
    if bh_path.exists():
        with open(bh_path) as f:
            bh_list = json.load(f)
        for bh in bh_list:
            split_id = bh.get("split", "?")
            rows.append({
                "strategy": "buy_and_hold",
                "split": split_id,
                "oos_years": split_to_oos_years(split_id),
                "total_trades": int(bh.get("total_trades", 1)),
                "total_return_pct": float(bh.get("total_return_pct", 0.0)),
                "win_rate_pct": float(bh.get("win_rate_pct", 0.0)),
                "profit_factor": None,
                "max_drawdown_pct": float(bh.get("max_drawdown_pct", 0.0)),
                "sharpe_ratio": None,  # 사안 (가) — B&H 분석 제외
                "calmar_ratio": None,
            })

    return pd.DataFrame(rows)


def compute_pairwise_bootstrap(eval_root: Path, n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """5 모델 pairwise bootstrap × 양 끝점 분할 (1, Exp4) = C(5,2)×2 = 20 비교."""
    rows: list[dict] = []
    for split in BOOTSTRAP_TARGET_SPLITS:
        # 5 모델 trades.csv 한 번에 로드
        pnl_by_model: dict[str, np.ndarray] = {}
        for strat in MODEL_STRATEGIES:
            trades_path = _spec_dir(eval_root, strat, split) / "trades.csv"
            if not trades_path.exists():
                logger.warning("trades 없음: %s", trades_path)
                continue
            df = pd.read_csv(trades_path)
            pnl_by_model[strat] = df["pnl"].to_numpy(dtype=np.float64)

        models_loaded = list(pnl_by_model.keys())
        for a, b in combinations(models_loaded, 2):
            logger.info("[BOOTSTRAP] split=%s | %s vs %s | n_a=%d n_b=%d",
                        split, a, b, len(pnl_by_model[a]), len(pnl_by_model[b]))
            obs, p, ci_low, ci_high = bootstrap_pnl_diff(
                pnl_by_model[a], pnl_by_model[b], n=n, seed=seed,
            )
            rows.append({
                "split": split,
                "model_a": a,
                "model_b": b,
                "n_a": len(pnl_by_model[a]),
                "n_b": len(pnl_by_model[b]),
                "mean_pnl_diff": round(obs, 4),
                "p_value": round(p, 4),
                "ci_low_95": round(ci_low, 4),
                "ci_high_95": round(ci_high, 4),
                "significant_at_0.05": p < 0.05,
            })

    df = pd.DataFrame(rows)

    # BL-1-2 (사안 D 다): Multi-hypothesis 보정 (Bonferroni + FDR)
    if not df.empty:
        raw_p = df["p_value"].to_numpy(dtype=np.float64)
        bonf_p, bonf_reject = bonferroni_correction(raw_p, alpha=0.05)
        fdr_p, fdr_reject = fdr_correction(raw_p, alpha=0.05)
        df["p_value_bonferroni"] = np.round(bonf_p, 4)
        df["significant_at_0.05_bonf"] = bonf_reject
        df["p_value_fdr"] = np.round(fdr_p, 4)
        df["significant_at_0.05_fdr"] = fdr_reject

        n_raw = int(df["significant_at_0.05"].sum())
        n_bonf = int(bonf_reject.sum())
        n_fdr = int(fdr_reject.sum())
        logger.info(
            "Multi-hypothesis 보정 결과 (N=%d 비교): "
            "raw=%d / Bonferroni=%d / FDR=%d 유의 (alpha=0.05)",
            len(df), n_raw, n_bonf, n_fdr,
        )

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="evaluate_models 결과 분석 (Phase E-2-4)")
    parser.add_argument("--eval-date", required=True, help="대상 폴더 suffix (예: 260503_baseline)")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()

    eval_root = REPORT_BASE / f"eval_{args.eval_date}"
    if not eval_root.exists():
        logger.error("폴더 없음: %s", eval_root)
        return 2

    if not args.bootstrap_only:
        logger.info("=" * 60)
        logger.info("Step 1: 매트릭스 수집 (30 specs + macross 6 + B&H 6)")
        df_metrics = collect_metrics_matrix(eval_root)
        out = eval_root / "analysis_metrics.csv"
        df_metrics.to_csv(out, index=False)
        logger.info("저장: %s (%d rows)", out, len(df_metrics))

    if not args.metrics_only:
        logger.info("=" * 60)
        logger.info("Step 2: 5 모델 pairwise bootstrap (분할 1, Exp4)")
        df_bs = compute_pairwise_bootstrap(eval_root, n=args.bootstrap_n, seed=args.seed)
        out = eval_root / "bootstrap_pvalues.csv"
        df_bs.to_csv(out, index=False)
        logger.info("저장: %s (%d rows)", out, len(df_bs))

    logger.info("=" * 60)
    logger.info("완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
