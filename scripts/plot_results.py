"""evaluate_models 결과 시각화 (Phase E-2-4 Step 3, 옵션 B).

분할별 1 PNG × 6 (5 모델 + macross + B&H equity overlay) + BTC 가격 보조축.

각 PNG:
  - 메인 plot (16x9): 7 곡선 equity_curve overlay (log scale Y축)
    - ml_lightgbm/ml_xgboost/dl_lstm/dl_transformer/rl_ppo (실선)
    - example_macross (회색 점선)
    - buy_and_hold (검정 점선, BTC 가격으로 시뮬레이션: initial × price/first_price)
  - 보조 Y축 (오른쪽): BTC 1d 가격 (lightcoral, 시장 흐름 비교)
  - Initial balance 가로선 ($10,000, 회색 점선)
  - 제목: Equity Overlay — Split {split_id} ({oos_start} ~ {oos_end})

Usage:
    python scripts/plot_results.py --eval-date 260503_baseline
    python scripts/plot_results.py --eval-date 260503_baseline --split 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_BASE = Path("data/backtest_reports/00_Working")
BTC_1D_CSV = Path("data/candles/BTC_USDT_USDT_1d.csv")

SPLIT_IDS = ["1", "A", "B", "Exp2", "Exp3", "Exp4"]
MODEL_STRATEGIES = ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer", "rl_ppo"]

# evaluate_models.py SPLIT_DEFINITIONS와 일관 — OOS 기간 매핑
SPLIT_OOS = {
    "1": ("2025-01-01", "2025-12-31"),
    "A": ("2023-01-01", "2023-12-31"),
    "B": ("2024-01-01", "2024-12-31"),
    "Exp2": ("2024-01-01", "2025-12-31"),
    "Exp3": ("2023-01-01", "2025-12-31"),
    "Exp4": ("2022-01-01", "2025-12-31"),
}

MODEL_COLORS = {
    "ml_lightgbm": "tab:blue",
    "ml_xgboost": "tab:green",
    "dl_lstm": "tab:purple",
    "dl_transformer": "tab:orange",
    "rl_ppo": "tab:red",
}


def _load_equity(cfg_dir: Path) -> pd.DataFrame | None:
    p = cfg_dir / "equity_curve.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _load_btc_1d(start: str, end: str) -> pd.DataFrame | None:
    if not BTC_1D_CSV.exists():
        return None
    btc = pd.read_csv(BTC_1D_CSV, parse_dates=["timestamp"], index_col="timestamp")
    btc.index = pd.to_datetime(btc.index, utc=True)
    btc = btc.loc[
        (btc.index >= pd.Timestamp(start, tz="UTC"))
        & (btc.index <= pd.Timestamp(end, tz="UTC"))
    ]
    return btc if not btc.empty else None


def _simulate_bh(btc_1d: pd.DataFrame, initial: float = 10000.0) -> pd.Series:
    """B&H equity 시뮬레이션 — initial × (close / first_close)."""
    first_close = float(btc_1d["close"].iloc[0])
    return (btc_1d["close"] / first_close) * initial


def plot_split(eval_root: Path, split_id: str, initial: float = 10000.0) -> Path | None:
    oos_start, oos_end = SPLIT_OOS[split_id]
    out_path = eval_root / f"equity_overlay_{split_id}.png"

    fig, ax = plt.subplots(figsize=(16, 9))

    # --- 5 모델 equity overlay ---
    for strat in MODEL_STRATEGIES:
        eq = _load_equity(eval_root / f"{strat}_{split_id}" / strat)
        if eq is None:
            logger.warning("equity 없음: %s split=%s", strat, split_id)
            continue
        ax.plot(eq.index, eq["equity"], color=MODEL_COLORS[strat], linewidth=1.2, label=strat)

    # --- macross (회색 점선) ---
    macross_dir = eval_root / f"macross_{split_id}" / f"_eval_macross_{split_id}"
    eq_mac = _load_equity(macross_dir)
    if eq_mac is not None:
        ax.plot(eq_mac.index, eq_mac["equity"], color="tab:gray",
                linestyle="--", linewidth=1.2, label="example_macross")

    # --- B&H 시뮬레이션 (검정 점선) ---
    btc_1d = _load_btc_1d(oos_start, oos_end)
    if btc_1d is not None:
        bh_eq = _simulate_bh(btc_1d, initial)
        ax.plot(bh_eq.index, bh_eq, color="black", linestyle="--",
                linewidth=1.5, label="buy_and_hold (BTC simulation)")

    # --- Initial 가로선 ---
    ax.axhline(y=initial, color="gray", linestyle=":", alpha=0.5,
               label=f"Initial ${initial:,.0f}")

    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($, log scale)", color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # --- 보조 Y축: BTC 가격 (옵션 B) ---
    if btc_1d is not None:
        ax2 = ax.twinx()
        ax2.plot(btc_1d.index, btc_1d["close"], color="lightcoral",
                 linewidth=1.0, alpha=0.7)
        ax2.set_ylabel("BTC Price ($)", color="lightcoral")
        ax2.tick_params(axis="y", labelcolor="lightcoral")

    plt.title(
        f"Equity Overlay — Split {split_id} ({oos_start} ~ {oos_end})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("저장: %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="evaluate_models 결과 시각화 (Phase E-2-4 Step 3)")
    parser.add_argument("--eval-date", required=True, help="대상 폴더 suffix (예: 260503_baseline)")
    parser.add_argument("--split", choices=SPLIT_IDS, help="특정 분할만 (생략 시 6 분할 모두)")
    args = parser.parse_args()

    eval_root = REPORT_BASE / f"eval_{args.eval_date}"
    if not eval_root.exists():
        logger.error("폴더 없음: %s", eval_root)
        return 2

    targets = [args.split] if args.split else SPLIT_IDS
    for split in targets:
        plot_split(eval_root, split)

    logger.info("완료: %d PNG 생성", len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
