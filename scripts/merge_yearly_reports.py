"""연도별 분할 백테스트 리포트를 복리 기준으로 통합하��� 스크립트.

Usage:
    python scripts/merge_yearly_reports.py [--tag TAG] [--initial-balance N]

Examples:
    python scripts/merge_yearly_reports.py
    python scripts/merge_yearly_reports.py --tag 260418
    python scripts/merge_yearly_reports.py --tag 260418 --initial-balance 10000

동작:
  1. data/backtest_reports/00_Working/ 에서 TAG에 매칭되는 연도별 리포트를 시간순 정렬
  2. trades.csv를 이어붙이되, PnL/size를 복리 스케일링 (연도별 $10K 리셋 → 연속 자본)
  3. equity_curve.csv를 복리 기준으로 연결 (이전 연도 final_balance를 다음 연도 시작으로)
  4. 통합 metrics를 재계산
  5. 통합 equity_curve.png 생성
  6. 결과를 00_Working/TAG_backtest_MERGE_START_END_config/ 에 저장
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


REPORT_WORKING = "data/backtest_reports/00_Working"


def find_yearly_dirs(tag: str, config_name: str) -> list[Path]:
    """태그+config 이름에 매칭되는 연도별 리포트 디렉토리를 시간순으로 반환.

    디렉토리 형식: {tag}_backtest_{start}_{end}_{config_name}
    MERGE 통합 리포트 (이름에 'MERGE' 포함)는 제외.
    """
    base = Path(REPORT_WORKING)
    if not base.exists():
        print(f"[ERROR] {base} 디렉토리가 없습니다.")
        sys.exit(1)

    pattern = re.compile(
        rf"^{re.escape(tag)}_backtest_(\d{{4}})-(\d{{2}})-(\d{{2}})_(\d{{4}})-(\d{{2}})-(\d{{2}})_{re.escape(config_name)}$"
    )
    dirs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and pattern.match(d.name):
            m = pattern.match(d.name)
            start_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            dirs.append((start_date, d))

    dirs.sort(key=lambda x: x[0])
    return [d for _, d in dirs]


def load_report_data(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """리포트 디렉토리에서 trades, equity_curve, metrics 로드.

    서브디렉토리 탐색 순서:
      1. report_dir 안의 첫 번째 서브디렉토리 (trades.csv 포함)
      2. report_dir 자체 (서브디렉토리 없는 경우)
    """
    config_dir = report_dir
    for child in sorted(report_dir.iterdir()):
        if child.is_dir() and (child / "trades.csv").exists():
            config_dir = child
            break

    trades_path = config_dir / "trades.csv"
    equity_path = config_dir / "equity_curve.csv"
    metrics_path = config_dir / "metrics.json"

    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"]) if trades_path.exists() else pd.DataFrame()
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"], index_col="timestamp") if equity_path.exists() else pd.DataFrame()
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return trades, equity, metrics


def merge_reports(dirs: list[Path], initial_balance: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """연도별 리포트를 복리 기준으로 통합."""
    all_trades = []
    all_equity = []
    carry_balance = initial_balance  # 이�� 연도 마감 잔고

    yearly_summaries = []

    for i, report_dir in enumerate(dirs):
        trades, equity, metrics = load_report_data(report_dir)
        year_label = report_dir.name.split("_")[2][:4]  # e.g., "2020"

        m = metrics.get("integrated", metrics)
        year_initial = m.get("initial_balance", 10000.0)
        year_final = m.get("final_balance", year_initial)

        # 스케일링 팩터: 연도별 $10K 기준 → 복리 연속 자본 기준
        scale = carry_balance / year_initial

        # trades 스케일링
        if not trades.empty:
            trades_scaled = trades.copy()
            trades_scaled["pnl"] = trades["pnl"] * scale
            trades_scaled["size"] = trades["size"] * scale
            # pnl_pct는 비율이므로 그대로 유지
            all_trades.append(trades_scaled)

        # equity curve 스케일링
        if not equity.empty:
            eq_scaled = equity.copy()
            eq_scaled["balance"] = equity["balance"] * scale
            eq_scaled["equity"] = equity["equity"] * scale
            all_equity.append(eq_scaled)

        # 연도 요약
        yearly_summaries.append({
            "year": year_label,
            "dir": report_dir.name,
            "scale": round(scale, 4),
            "carry_in": round(carry_balance, 2),
            "year_return_pct": round(m.get("total_return_pct", 0), 2),
            "carry_out": round(carry_balance * (year_final / year_initial), 2),
        })

        # 다음 연도 시작 잔고 = 이 연도 마감 잔고 (복리)
        carry_balance = carry_balance * (year_final / year_initial)

    # 통합 trades
    merged_trades = pd.concat(all_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    # 통합 equity curve (겹치는 타임스탬프 제거: 이후 연도 우선)
    merged_equity = pd.concat(all_equity)
    merged_equity = merged_equity[~merged_equity.index.duplicated(keep="last")]
    merged_equity.sort_index(inplace=True)

    # 통합 metrics 계산
    merged_metrics = compute_merged_metrics(merged_trades, merged_equity, initial_balance, carry_balance)
    merged_metrics["yearly_summaries"] = yearly_summaries

    return merged_trades, merged_equity, merged_metrics


def compute_merged_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_balance: float,
    final_balance: float,
) -> dict:
    """통합 데이터에서 성과 지표 재계산."""
    total = len(trades)
    if total == 0:
        return {"error": "No trades"}

    winning = trades[trades["pnl"] > 0]
    losing = trades[trades["pnl"] <= 0]

    win_rate = len(winning) / total * 100
    total_pnl = trades["pnl"].sum()
    gross_profit = winning["pnl"].sum() if not winning.empty else 0
    gross_loss = abs(losing["pnl"].sum()) if not losing.empty else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # MDD from equity curve
    max_dd = 0
    max_dd_pct = 0
    if not equity.empty:
        eq = equity["equity"]
        peak = eq.expanding().max()
        dd = peak - eq
        dd_pct = dd / peak * 100
        max_dd = float(dd.max())
        max_dd_pct = float(dd_pct.max())

    # Sharpe (annualized, daily)
    sharpe = 0
    if not equity.empty and len(equity) > 1:
        daily = equity["equity"].resample("1D").last().dropna()
        if len(daily) > 1:
            returns = daily.pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * (365 ** 0.5)

    # Average duration
    avg_duration = None
    if "entry_time" in trades and "exit_time" in trades:
        durations = trades["exit_time"] - trades["entry_time"]
        avg_duration = str(durations.mean())

    # Strategy split (신규 컬럼 strategy_name; 옛 컬럼 owner 도 fallback)
    strategy_split = {}
    strategy_col = (
        "strategy_name" if "strategy_name" in trades.columns
        else ("owner" if "owner" in trades.columns else None)
    )
    if strategy_col is not None:
        for name in trades[strategy_col].unique():
            sub = trades[trades[strategy_col] == name]
            w = sub[sub["pnl"] > 0]
            l = sub[sub["pnl"] <= 0]
            gp = w["pnl"].sum() if not w.empty else 0
            gl = abs(l["pnl"].sum()) if not l.empty else 0
            strategy_split[name] = {
                "trades": len(sub),
                "win_rate_pct": round(len(w) / len(sub) * 100, 1) if len(sub) > 0 else 0,
                "pnl": round(sub["pnl"].sum(), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            }

    # Exit reason split
    exit_split = {}
    if "exit_reason" in trades.columns:
        for reason in trades["exit_reason"].unique():
            sub = trades[trades["exit_reason"] == reason]
            w = sub[sub["pnl"] > 0]
            l = sub[sub["pnl"] <= 0]
            gp = w["pnl"].sum() if not w.empty else 0
            gl = abs(l["pnl"].sum()) if not l.empty else 0
            exit_split[reason] = {
                "trades": len(sub),
                "win_rate_pct": round(len(w) / len(sub) * 100, 1) if len(sub) > 0 else 0,
                "pnl": round(sub["pnl"].sum(), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            }

    # Direction split
    dir_split = {}
    if "side" in trades.columns:
        for side in trades["side"].unique():
            sub = trades[trades["side"] == side]
            w = sub[sub["pnl"] > 0]
            l = sub[sub["pnl"] <= 0]
            gp = w["pnl"].sum() if not w.empty else 0
            gl = abs(l["pnl"].sum()) if not l.empty else 0
            dir_split[side] = {
                "trades": len(sub),
                "win_rate_pct": round(len(w) / len(sub) * 100, 1) if len(sub) > 0 else 0,
                "pnl": round(sub["pnl"].sum(), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            }

    return {
        "integrated": {
            "total_trades": total,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((final_balance - initial_balance) / initial_balance * 100, 2),
            "avg_win": round(winning["pnl"].mean(), 2) if not winning.empty else 0,
            "avg_loss": round(abs(losing["pnl"].mean()), 2) if not losing.empty else 0,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_duration": avg_duration,
            "initial_balance": initial_balance,
            "final_balance": round(final_balance, 2),
        },
        "by_strategy_name": strategy_split,
        "by_exit_reason": exit_split,
        "by_direction": dir_split,
    }


def plot_equity(equity: pd.DataFrame, initial_balance: float, save_path: str):
    """통합 equity curve + drawdown 차트 생성."""
    if equity.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(equity.index, equity["equity"], color="blue", linewidth=1, label="Equity")
    axes[0].axhline(y=initial_balance, color="gray", linestyle="--", alpha=0.5, label=f"Initial ${initial_balance:,.0f}")
    axes[0].set_title("Merged Equity Curve (Compounded) vs BTC Price")
    axes[0].set_ylabel("Equity ($)", color="blue")
    axes[0].tick_params(axis="y", labelcolor="blue")
    axes[0].grid(True, alpha=0.3)

    # BTC overlay
    try:
        btc_csv = os.path.join("data", "candles", "BTC_USDT_USDT_1d.csv")
        if os.path.exists(btc_csv):
            btc = pd.read_csv(btc_csv, parse_dates=["timestamp"])
            btc.set_index("timestamp", inplace=True)
            btc = btc.loc[(btc.index >= equity.index.min()) & (btc.index <= equity.index.max())]
            if not btc.empty:
                ax2 = axes[0].twinx()
                ax2.plot(btc.index, btc["close"], color="orange", linewidth=1, alpha=0.7, label="BTC")
                ax2.set_ylabel("BTC Price ($)", color="orange")
                ax2.tick_params(axis="y", labelcolor="orange")
    except Exception:
        pass

    axes[0].legend(loc="upper left")

    # Drawdown
    eq = equity["equity"]
    peak = eq.expanding().max()
    dd_pct = (peak - eq) / peak * 100
    axes[1].fill_between(dd_pct.index, 0, dd_pct, color="red", alpha=0.3)
    axes[1].set_title("Drawdown (%)")
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {save_path}")


def print_summary(metrics: dict):
    """터미널에 통합 리포트 출력."""
    m = metrics["integrated"]

    print("\n" + "=" * 70)
    print("         MERGED BACKTEST REPORT (COMPOUNDED)")
    print("=" * 70)
    for k, v in m.items():
        if k in ("initial_balance", "final_balance"):
            print(f"  {k.replace('_', ' ').title():.<40} ${v:,.2f}")
        elif k in ("total_pnl", "gross_profit", "gross_loss", "avg_win", "avg_loss", "max_drawdown"):
            print(f"  {k.replace('_', ' ').title():.<40} ${v:,.2f}")
        elif "pct" in k:
            print(f"  {k.replace('_', ' ').title():.<40} {v}%")
        else:
            print(f"  {k.replace('_', ' ').title():.<40} {v}")
    print("=" * 70)

    # Strategy split
    if "by_strategy_name" in metrics and metrics["by_strategy_name"]:
        print("\n  --- By Strategy ---")
        for name, data in metrics["by_strategy_name"].items():
            print(f"  {name:>20s}: {data['trades']}건  WR {data['win_rate_pct']}%  PnL ${data['pnl']:,.2f}  PF {data['profit_factor']}")

    # Exit reason split
    if "by_exit_reason" in metrics and metrics["by_exit_reason"]:
        print("\n  --- By Exit Reason ---")
        for reason, data in sorted(metrics["by_exit_reason"].items(), key=lambda x: -x[1]["trades"]):
            print(f"  {reason:>20s}: {data['trades']}건  WR {data['win_rate_pct']}%  PnL ${data['pnl']:,.2f}  PF {data['profit_factor']}")

    # Direction split
    if "by_direction" in metrics and metrics["by_direction"]:
        print("\n  --- By Direction ---")
        for side, data in metrics["by_direction"].items():
            print(f"  {side:>20s}: {data['trades']}건  WR {data['win_rate_pct']}%  PnL ${data['pnl']:,.2f}  PF {data['profit_factor']}")

    # Yearly compounding summary
    if "yearly_summaries" in metrics:
        print("\n  --- Yearly Compounding ---")
        print(f"  {'Year':>6s}  {'Carry In':>12s}  {'Return%':>8s}  {'Carry Out':>12s}  {'Scale':>7s}")
        for ys in metrics["yearly_summaries"]:
            print(f"  {ys['year']:>6s}  ${ys['carry_in']:>11,.2f}  {ys['year_return_pct']:>7.2f}%  ${ys['carry_out']:>11,.2f}  {ys['scale']:>7.4f}")

    print()


def list_available(base: Path) -> list[tuple[str, str]]:
    """00_Working 내 연도별 리포트에서 사용 가능한 (tag, config_name) 조합 목록 반환."""
    pattern = re.compile(r"^(\d{6})_backtest_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(.+)$")
    combos: dict[tuple[str, str], int] = {}
    for d in sorted(base.iterdir()):
        if d.is_dir():
            m = pattern.match(d.name)
            if m and "MERGE" not in d.name:
                key = (m.group(1), m.group(2))
                combos[key] = combos.get(key, 0) + 1
    return [(tag, cfg, cnt) for (tag, cfg), cnt in sorted(combos.items())]


def main():
    parser = argparse.ArgumentParser(description="연도별 분할 백테스트 리포트 통합 (복리)")
    parser.add_argument("--tag", required=True, help="리포트 날짜 태그 (예: 260418)")
    parser.add_argument("--config-name", required=True, help="config 이름 (예: config, config_aggressive)")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="통합 시작 잔고 (기본: 10000)")
    args = parser.parse_args()

    tag = args.tag
    config_name = args.config_name

    dirs = find_yearly_dirs(tag, config_name)
    if len(dirs) < 2:
        print(f"\n[ERROR] tag='{tag}', config='{config_name}'에 매칭되는 연도별 리포트가 {len(dirs)}개뿐입니다. (최소 2개 필요)")
        base = Path(REPORT_WORKING)
        if base.exists():
            available = list_available(base)
            if available:
                print("\n  사용 가능한 조합:")
                print(f"  {'Tag':>8s}  {'Config Name':<40s}  {'Years':>5s}")
                print(f"  {'---':>8s}  {'---':<40s}  {'---':>5s}")
                for t, c, n in available:
                    print(f"  {t:>8s}  {c:<40s}  {n:>5d}")
        sys.exit(1)

    print(f"\n  Found {len(dirs)} yearly reports (tag={tag}, config={config_name}):")
    for d in dirs:
        print(f"    - {d.name}")

    # 통합
    merged_trades, merged_equity, merged_metrics = merge_reports(dirs, args.initial_balance)

    # 출력 디렉토리
    # 디렉토리명에서 날짜 추출: {tag}_backtest_{start}_{end}_{config_name}
    date_pat = re.compile(r"_backtest_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_")
    m_first = date_pat.search(dirs[0].name)
    m_last = date_pat.search(dirs[-1].name)
    start_date = m_first.group(1) if m_first else "unknown"
    end_date = m_last.group(2) if m_last else "unknown"
    today = datetime.now().strftime("%y%m%d")
    out_name = f"{today}_backtest_MERGE_{start_date}_{end_date}_{config_name}"
    out_dir = os.path.join(REPORT_WORKING, out_name, config_name)
    os.makedirs(out_dir, exist_ok=True)

    # 저장
    merged_trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    print(f"  Trades saved: {os.path.join(out_dir, 'trades.csv')} ({len(merged_trades)} trades)")

    merged_equity.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    print(f"  Equity curve saved: {os.path.join(out_dir, 'equity_curve.csv')}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(merged_metrics, f, indent=2, default=str)
    print(f"  Metrics saved: {os.path.join(out_dir, 'metrics.json')}")

    # config snapshot 복사 (첫 번째 연도 것 사용)
    first_snapshot = None
    for child in sorted(dirs[0].iterdir()):
        if child.is_dir() and (child / "config_snapshot.yaml").exists():
            first_snapshot = child / "config_snapshot.yaml"
            break
    if first_snapshot and first_snapshot.exists():
        with open(first_snapshot) as f:
            config = yaml.safe_load(f)
        with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # 차트
    plot_equity(merged_equity, args.initial_balance, os.path.join(out_dir, "equity_curve.png"))

    # 터미널 출력
    print_summary(merged_metrics)

    print(f"  Output: {os.path.join(REPORT_WORKING, out_name)}")


if __name__ == "__main__":
    main()
