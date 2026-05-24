"""OKX fetch_my_trades 응답 직접 진단 — BLE-6-1 sync 매칭 실패 원인 추적.

실행: python scripts/diagnose_okx_fills.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config  # noqa: E402


async def main() -> None:
    import ccxt.async_support as ccxt
    cfg = load_config("config/ensemble.yaml")
    exch_cfg = cfg["exchange"]
    # config_loader 가 .env → exchange.{api_key, secret, passphrase} 로 주입
    exchange = ccxt.okx({
        "apiKey": exch_cfg.get("api_key"),
        "secret": exch_cfg.get("secret"),
        "password": exch_cfg.get("passphrase"),
        "options": {"defaultType": "swap"},
    })
    symbol = exch_cfg.get("symbol", "BTC/USDT:USDT")
    # 2026-05-06 12:14:01 UTC = 가장 오래된 미sync trade ts (12:15:01) - 60초
    since_ms = int(
        datetime(2026, 5, 6, 12, 14, 1, tzinfo=timezone.utc).timestamp() * 1000
    )

    print(f"=== fetch_my_trades(symbol={symbol}, since={since_ms}, limit=200) ===")
    try:
        trades = await exchange.fetch_my_trades(symbol, since=since_ms, limit=200)
    except Exception as e:
        print(f"ERROR: {e}")
        await exchange.close()
        return

    print(f"Total fills: {len(trades)}")
    if not trades:
        print("(빈 응답 — OKX 가 since 무시하거나 권한 영역 점검 필요)")
        await exchange.close()
        return

    # 시계열 정렬
    trades_sorted = sorted(trades, key=lambda t: t.get("timestamp") or 0)
    print(f"\nfirst fill timestamp: {trades_sorted[0].get('timestamp')} "
          f"({trades_sorted[0].get('datetime')})")
    print(f"last fill timestamp:  {trades_sorted[-1].get('timestamp')} "
          f"({trades_sorted[-1].get('datetime')})")

    # 첫 fill 의 전체 필드 (구조 분석용)
    print(f"\n=== 첫 fill 전체 필드 (구조 진단) ===")
    print(json.dumps(trades_sorted[0], indent=2, default=str))

    # 마지막 fill (Trade 16 close 영역일 가능성)
    print(f"\n=== 마지막 fill 전체 필드 ===")
    print(json.dumps(trades_sorted[-1], indent=2, default=str))

    # 핵심 필드 sample (5건)
    print(f"\n=== 핵심 필드 sample (시간 ASC, 처음 5건) ===")
    for t in trades_sorted[:5]:
        print(json.dumps({
            "order": t.get("order"),
            "id": t.get("id"),
            "timestamp": t.get("timestamp"),
            "datetime": t.get("datetime"),
            "side": t.get("side"),
            "amount": t.get("amount"),
            "price": t.get("price"),
            "fee": t.get("fee"),
            "reduceOnly_raw": t.get("reduceOnly"),
            "info.reduceOnly": (t.get("info") or {}).get("reduceOnly"),
            "info_keys": list((t.get("info") or {}).keys())[:10],
        }, default=str))

    # order_id 별 그룹 수
    by_order: dict = {}
    for t in trades_sorted:
        oid = t.get("order")
        by_order.setdefault(oid, []).append(t)
    print(f"\n=== order_id 별 그룹 ({len(by_order)}개 unique orders) ===")
    for oid, fills in list(by_order.items())[:10]:
        total = sum(float(f.get("amount", 0) or 0) for f in fills)
        side = fills[0].get("side")
        ts = fills[0].get("datetime")
        print(f"  {oid}: {len(fills)} fills, side={side}, "
              f"total_amount={total:.6f}, first_ts={ts}")

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
