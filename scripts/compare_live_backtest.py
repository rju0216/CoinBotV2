"""BLE-6-2: 라이브 vs 백테 정합성 비교 (trade-by-trade + aggregate).

입금(2026-05-23)으로 size 가 점프하므로 입금 시점 2구간으로 분할 비교한다.
구간별 백테(initial = 라이브 해당 구간 시작 balance)를 라이브와 매칭한다.

매칭: 같은 side + entry 시각 ±MATCH_WINDOW_MIN 분 내 가장 가까운 백테 거래 (1:1).
정량: entry/exit 가격 slippage, pnl, pnl_pct, fee율, funding 차이.
      가격·비율 항목은 size/initial 무관 → 입금 영향 없음.

baseline 재현용 (V'=가 정적). 사용법:
  python scripts/compare_live_backtest.py \
    --live-db data/coinbot_live.db \
    --deposit-ts 2026-05-23T05:34:40+00:00 \
    --bt-seg1 <구간1 백테 결과 디렉토리>/ensemble \
    --bt-seg2 <구간2 백테 결과 디렉토리>/ensemble \
    --out docs/02_Limitations/LIVE_BACKTEST_PARITY_260615.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

import pandas as pd

MATCH_WINDOW_MIN = 15  # §8.5 검증 기준: entry 시각 ±15분


def _to_dt(v) -> datetime:
    """ISO str / pandas Timestamp / datetime → tz-aware datetime (UTC)."""
    if isinstance(v, datetime):
        dt = v
    else:
        dt = pd.to_datetime(v, utc=True).to_pydatetime()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_live_trades(db_path: str, deposit_ts: str) -> dict[str, list[dict]]:
    """라이브 closed trades 를 입금 시점 기준 seg1(입금 전)/seg2(입금 후)로 분할."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, timestamp, side, size, entry_price, exit_price, "
        "pnl, pnl_pct, trading_fee, funding_fee, exit_reason, closed_at "
        "FROM trades WHERE status='closed' ORDER BY timestamp"
    ).fetchall()
    con.close()
    dep = _to_dt(deposit_ts)
    seg1, seg2 = [], []
    for r in rows:
        d = dict(r)
        d["entry_dt"] = _to_dt(d["timestamp"])
        (seg1 if d["entry_dt"] < dep else seg2).append(d)
    return {"seg1": seg1, "seg2": seg2}


def load_bt_trades(bt_dir: str) -> list[dict]:
    """백테 trades.csv 로드."""
    p = Path(bt_dir) / "trades.csv"
    df = pd.read_csv(p)
    out = []
    for _, row in df.iterrows():
        if str(row.get("status")) != "closed":
            continue
        d = row.to_dict()
        d["entry_dt"] = _to_dt(row["entry_time"])
        out.append(d)
    return out


def match_segment(live: list[dict], bt: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """side + entry 시각 ±MATCH_WINDOW_MIN 매칭 (1:1, greedy nearest).

    반환: (matched_pairs, live_unmatched, bt_unmatched)
    """
    used_bt: set[int] = set()
    pairs, live_un = [], []
    for lt in live:
        best, best_gap = None, None
        for i, bt_t in enumerate(bt):
            if i in used_bt:
                continue
            if str(bt_t["side"]) != str(lt["side"]):
                continue
            gap = abs((lt["entry_dt"] - bt_t["entry_dt"]).total_seconds())
            if gap <= MATCH_WINDOW_MIN * 60 and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is not None:
            used_bt.add(best)
            pairs.append({"live": lt, "bt": bt[best], "gap_sec": best_gap})
        else:
            live_un.append(lt)
    bt_un = [bt[i] for i in range(len(bt)) if i not in used_bt]
    return pairs, live_un, bt_un


def _stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def diff_pairs(pairs: list[dict]) -> dict:
    """매칭쌍별 차이 정량. 가격·비율은 size 무관 (입금 영향 없음)."""
    entry_slip_abs, entry_slip_pct, exit_slip_abs = [], [], []
    pnl_pct_diff, fee_rate_live, fee_rate_bt, funding_live = [], [], [], []
    rows = []
    for p in pairs:
        lv, bt = p["live"], p["bt"]
        side = str(lv["side"])
        sign = 1 if side == "long" else -1
        # entry slippage: 라이브 진입가 - 백테 진입가 (불리 방향 부호 통일: +=라이브가 더 불리)
        e_abs = (lv["entry_price"] - bt["entry_price"]) * sign
        e_pct = e_abs / bt["entry_price"] * 100 if bt["entry_price"] else 0.0
        x_abs = (lv["exit_price"] - bt["exit_price"]) * (-sign) if lv.get("exit_price") and bt.get("exit_price") else None
        ppd = (lv["pnl_pct"] or 0) - (bt.get("pnl_pct") or 0)
        ln = lv["entry_price"] * lv["size"]
        bn = bt["entry_price"] * bt["size"]
        frl = (lv["trading_fee"] or 0) / ln * 100 if ln else 0.0
        frb = (bt.get("trading_fee") or 0) / bn * 100 if bn else 0.0
        entry_slip_abs.append(e_abs); entry_slip_pct.append(e_pct)
        if x_abs is not None:
            exit_slip_abs.append(x_abs)
        pnl_pct_diff.append(ppd); fee_rate_live.append(frl); fee_rate_bt.append(frb)
        funding_live.append(lv["funding_fee"] or 0)
        rows.append({
            "live_id": lv["id"], "bt_id": bt.get("id"), "side": side,
            "gap_sec": round(p["gap_sec"]),
            "entry_live": lv["entry_price"], "entry_bt": bt["entry_price"],
            "entry_slip_pct": round(e_pct, 4),
            "exit_live": lv.get("exit_price"), "exit_bt": bt.get("exit_price"),
            "pnl_live": lv["pnl"], "pnl_bt": bt.get("pnl"),
            "pnl_pct_live": lv["pnl_pct"], "pnl_pct_bt": bt.get("pnl_pct"),
            "funding_live": lv["funding_fee"], "exit_reason_live": lv["exit_reason"],
            "exit_reason_bt": bt.get("exit_reason"),
        })
    return {
        "entry_slippage_pct": _stats(entry_slip_pct),
        "entry_slippage_abs": _stats(entry_slip_abs),
        "exit_slippage_abs": _stats(exit_slip_abs),
        "pnl_pct_diff": _stats(pnl_pct_diff),
        "fee_rate_live_pct": _stats(fee_rate_live),
        "fee_rate_bt_pct": _stats(fee_rate_bt),
        "funding_live_per_trade": _stats(funding_live),
        "rows": rows,
    }


def agg(trades: list[dict], pnl_key: str = "pnl") -> dict:
    pnls = [t.get(pnl_key) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        "n": len(trades),
        "sum_pnl": round(sum(pnls), 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
    }


def run(live_db: str, deposit_ts: str, bt_seg1: str, bt_seg2: str) -> dict:
    live = load_live_trades(live_db, deposit_ts)
    report = {"deposit_ts": deposit_ts, "match_window_min": MATCH_WINDOW_MIN, "segments": {}}
    for seg, bt_dir in [("seg1", bt_seg1), ("seg2", bt_seg2)]:
        lv = live[seg]
        bt = load_bt_trades(bt_dir)
        pairs, lv_un, bt_un = match_segment(lv, bt)
        match_rate = len(pairs) / len(lv) * 100 if lv else 0.0
        report["segments"][seg] = {
            "live_count": len(lv), "bt_count": len(bt),
            "matched": len(pairs), "match_rate_pct": round(match_rate, 1),
            "live_unmatched": [t["id"] for t in lv_un],
            "bt_unmatched": [t.get("id") for t in bt_un],
            "aggregate_live": agg(lv),
            "aggregate_bt": agg(bt),
            "trade_by_trade": diff_pairs(pairs),
        }
    return report


def _print_summary(rep: dict) -> None:
    print(f"=== BLE-6-2 라이브 vs 백테 정합성 (입금 ts={rep['deposit_ts']}) ===")
    for seg, s in rep["segments"].items():
        print(f"\n[{seg}] 라이브 {s['live_count']} / 백테 {s['bt_count']} "
              f"→ 매칭 {s['matched']} ({s['match_rate_pct']}%)")
        print(f"  aggregate live={s['aggregate_live']}  bt={s['aggregate_bt']}")
        tbt = s["trade_by_trade"]
        print(f"  entry slippage %: {tbt['entry_slippage_pct']}")
        print(f"  pnl_pct diff: {tbt['pnl_pct_diff']}")
        print(f"  fee rate live/bt %: {tbt['fee_rate_live_pct']} / {tbt['fee_rate_bt_pct']}")
        print(f"  funding/trade(live): {tbt['funding_live_per_trade']}")
        if s["live_unmatched"]:
            print(f"  ⚠ live 미매칭: {s['live_unmatched']}")
        if s["bt_unmatched"]:
            print(f"  ⚠ bt 미매칭: {s['bt_unmatched']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-db", default="data/coinbot_live.db")
    ap.add_argument("--deposit-ts", default="2026-05-23T05:34:40+00:00")
    ap.add_argument("--bt-seg1", required=True, help="구간1 백테 결과 디렉토리 (.../ensemble)")
    ap.add_argument("--bt-seg2", required=True, help="구간2 백테 결과 디렉토리 (.../ensemble)")
    ap.add_argument("--out", default=None, help="JSON 출력 경로 (선택)")
    args = ap.parse_args()
    rep = run(args.live_db, args.deposit_ts, args.bt_seg1, args.bt_seg2)
    _print_summary(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n→ 저장: {args.out}")


if __name__ == "__main__":
    main()
