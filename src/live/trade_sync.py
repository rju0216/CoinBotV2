"""TradeSyncer — DB trades 의 거래 수치를 OKX 실값으로 sync (BLE-6-1).

라이브 운영 중 close_position 직후 호출되어 미sync 모든 trade 를 batch 처리.
- 신규 거래: entry/exit_order_id 가 있으면 id 기반 정확 매칭
- 기존 거래 (id 없음): 시간/방향/size/reduceOnly 시간 매칭 fallback + OKX id 도 같이 저장

알고리즘:
1. paper 가드 (broker.is_live=False → skip)
2. data_store.get_unsynced_trades() — closed AND synced_at IS NULL
3. earliest_ts - 60s 기준 _fetch_all_fills pagination (100 fills/page × MAX_PAGES)
4. order_id 별 그룹화 + aggregate
   - amount: contracts × contract_size → BTC 변환 (I-BLE002 fix)
   - reduce_only: fillPnl 합 ≠ 0 (entry=0, exit≠0 — I-BLE002 fix)
5. 각 미sync trade 매칭 (id 우선 → 시간/방향/size + reduce_only fallback)
6. 매칭 성공 (entry+exit 둘 다) 시 entry/exit_price, trading_fee, pnl 재계산 + id 저장 + synced_at
7. 매칭 실패 시 WARNING + synced_at NULL 유지 (다음 시도)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


SIZE_TOLERANCE_BTC = 0.005          # N=가': 1 contract (0.01 BTC) 의 half, round 영향 안전
TIME_MARGIN_MS = 60_000              # I=가: ±60초 (fetch since + 매칭 tolerance 공통)
PAGE_LIMIT = 100                     # I-BLE002: OKX V5 /api/v5/trade/fills 한 page 한도
MAX_PAGES = 10                       # I-BLE002: pagination 안전망 (100 × 10 = 1000 fills)


async def sync_all_unsynced(
    broker: Any,
    data_store: Any,
    symbol: str,
    accounting_config: dict | None = None,
) -> dict:
    """미sync trade 일괄 처리. CoreEngine._close_with_funding 직후 호출 (라이브 전용).

    Args:
        broker: Broker (live/paper). broker.is_live + broker.executor.exchange 사용
        data_store: DataStore. get_unsynced_trades + update_synced_trade 사용
        symbol: 거래 symbol (예: "BTC/USDT:USDT")
        accounting_config: (미사용, 미래 확장 reserved)

    Returns:
        {synced_count, failed_count, errors: list[str]}
    """
    # K=가: paper 가드
    if not getattr(broker, "is_live", False):
        return {"synced_count": 0, "failed_count": 0, "errors": []}

    # 1. 미sync trade 조회
    try:
        unsynced = await data_store.get_unsynced_trades()
    except Exception as e:
        logger.warning("get_unsynced_trades 실패 — sync skip: %s", e)
        return {"synced_count": 0, "failed_count": 0, "errors": [f"get_unsynced: {e}"]}
    if not unsynced:
        return {"synced_count": 0, "failed_count": 0, "errors": []}

    # 2. earliest_ts 기준 OKX fills pagination fetch
    earliest_ts_ms = min(_parse_iso_ms(t["timestamp"]) for t in unsynced)
    since_ms = earliest_ts_ms - TIME_MARGIN_MS

    executor = broker.executor
    if executor is None or not hasattr(executor, "exchange"):
        return {"synced_count": 0, "failed_count": 0, "errors": ["broker.executor 없음"]}

    # I-BLE002: contract_size 추출 (LiveExecutor.initialize 시 OKX market 의 contractSize)
    contract_size = float(getattr(executor, "contract_size", 1.0) or 1.0)
    if contract_size <= 0:
        contract_size = 1.0

    try:
        okx_fills = await _fetch_all_fills(executor.exchange, symbol, since_ms)
    except Exception as e:
        logger.warning("fetch_my_trades 실패 — sync skip: %s", e)
        return {
            "synced_count": 0,
            "failed_count": len(unsynced),
            "errors": [f"fetch_my_trades: {e}"],
        }

    # 3. order_id 별 그룹화 + aggregate (다중 fill 대비)
    by_order = _group_and_aggregate(okx_fills, contract_size=contract_size)

    # 4. 각 미sync trade 매칭 + DB UPDATE
    results = {"synced_count": 0, "failed_count": 0, "errors": []}
    for db_trade in unsynced:
        try:
            entry, exit_ = _match_trade(db_trade, by_order)
            if entry and exit_:
                pnl_new = _recalc_pnl(
                    entry, exit_, db_trade["side"], db_trade.get("funding_fee", 0.0) or 0.0
                )
                await data_store.update_synced_trade(
                    trade_id=db_trade["id"],
                    entry_price=entry["price_avg"],
                    exit_price=exit_["price_avg"],
                    trading_fee=entry["fee_usdt"] + exit_["fee_usdt"],
                    pnl=pnl_new,
                    entry_order_id=entry["order_id"],
                    exit_order_id=exit_["order_id"],
                    synced_at=datetime.now(timezone.utc).isoformat(),
                )
                results["synced_count"] += 1
            else:
                # J=가: 매칭 실패 시 WARNING + synced_at NULL 유지 (다음 시도)
                # I-BLE004: 진단 정보 추가 — by_order 후보 검색 (어느 조건이 안 맞았는지)
                diag = _diagnose_match_failure(db_trade, by_order)
                msg = (
                    f"trade {db_trade['id']} 매칭 실패 "
                    f"(side={db_trade['side']}, size={db_trade['size']:.6f}, "
                    f"ts={db_trade['timestamp']})\n   {diag}"
                )
                logger.warning("Trade sync: %s", msg)
                results["failed_count"] += 1
                results["errors"].append(msg)
        except Exception as e:
            logger.warning("Trade sync trade %d exception: %s", db_trade.get("id"), e)
            results["failed_count"] += 1
            results["errors"].append(f"trade {db_trade.get('id')}: {e}")
    return results


def _parse_iso_ms(iso_str: str) -> int:
    """ISO timestamp → UTC ms."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


async def _fetch_all_fills(
    exchange: Any, symbol: str, since_ms: int,
) -> list[dict]:
    """I-BLE002: pagination 으로 since_ms 이후 모든 fill 누적.

    OKX V5 /api/v5/trade/fills 는 한 page 100 fills 한도. ccxt limit=200 요청해도
    100 만 반환. 다중 fill 큰 거래 (예: 24 fills 1 order) + 다수 거래 누적 시 잘림.

    알고리즘:
    - 매 page fetch 후 신규 fill (id 기준 dedup) 누적
    - 다음 page since = 마지막 fill ts (id dedup 이 중복 차단)
    - len(fills) < PAGE_LIMIT → 마지막 page
    - last_ts <= current_since → 무한 루프 차단 (모든 fill 같은 ts)
    - MAX_PAGES 안전망 도달 시 WARNING + 종료
    """
    all_fills: list[dict] = []
    seen_ids: set[str] = set()
    current_since = since_ms
    for page in range(MAX_PAGES):
        fills = await exchange.fetch_my_trades(
            symbol, since=current_since, limit=PAGE_LIMIT,
        )
        if not fills:
            break
        new_fills = [f for f in fills if f.get("id") not in seen_ids]
        if not new_fills:
            break   # 모두 중복 → 더 진행 안 됨
        all_fills.extend(new_fills)
        seen_ids.update(f["id"] for f in new_fills if f.get("id"))
        if len(fills) < PAGE_LIMIT:
            break   # 마지막 page
        last_ts = max(
            int(f["timestamp"]) for f in new_fills if f.get("timestamp")
        )
        if last_ts <= current_since:
            break   # 무한 루프 보호 (모든 신규 fill 이 같은 ts ≤ since)
        current_since = last_ts
    else:
        logger.warning(
            "Trade sync: MAX_PAGES=%d 도달 — 누적 %d fills, 누락 가능. "
            "since 분할 또는 MAX_PAGES 확대 고려.",
            MAX_PAGES, len(all_fills),
        )
    return all_fills


def _group_and_aggregate(
    okx_fills: list[dict], contract_size: float,
) -> dict[str, dict]:
    """order_id 별 fills 묶음 + aggregate (price avg / amount BTC 합 / fee 합 / ts / side / reduce_only).

    I-BLE002 fix:
    - amount 는 contracts 단위 (OKX fetch_my_trades 응답) → contract_size 곱해 BTC 변환 후 저장
    - reduce_only 는 fillPnl 합 ≠ 0 으로 판별 (OKX 응답에 reduceOnly 필드 부재)
      · entry order 의 모든 fill 은 fillPnl=0 → 합=0 → reduce_only=False
      · exit order 의 모든 fill 은 fillPnl≠0 → 합≠0 → reduce_only=True
    M=가: USDT fee 만 합산. 다른 currency 면 WARNING + 0 처리.
    """
    by_order_raw: dict[str, list[dict]] = defaultdict(list)
    for fill in okx_fills:
        oid = fill.get("order")
        if oid:
            by_order_raw[oid].append(fill)

    by_order: dict[str, dict] = {}
    for oid, fills in by_order_raw.items():
        total_amount_contracts = sum(float(f.get("amount", 0) or 0) for f in fills)
        if total_amount_contracts <= 0:
            continue
        total_amount_btc = total_amount_contracts * contract_size

        # M=가: USDT fee 만 합산
        fee_usdt = 0.0
        for f in fills:
            fee_obj = f.get("fee") or {}
            ccy = fee_obj.get("currency")
            cost = fee_obj.get("cost")
            if ccy == "USDT" and cost is not None:
                # OKX fee 는 음수 (지출), 시스템은 양수로 통일
                fee_usdt += abs(float(cost))
            elif ccy and ccy != "USDT":
                logger.warning(
                    "Trade sync: fee currency=%s (cost=%s) order=%s — 0 처리",
                    ccy, cost, oid,
                )

        # I-BLE002: fillPnl 합산 기반 reduce_only 판별
        total_fill_pnl = sum(
            float((f.get("info") or {}).get("fillPnl", 0) or 0)
            for f in fills
        )
        reduce_only = abs(total_fill_pnl) > 1e-9
        # price_avg = volume-weighted avg (contracts 가중치 — BTC 가중치와 비율 동일)
        price_avg = (
            sum(float(f["price"]) * float(f["amount"]) for f in fills)
            / total_amount_contracts
        )
        ts_ms = min(int(f["timestamp"]) for f in fills if f.get("timestamp"))
        by_order[oid] = {
            "order_id": oid,
            "price_avg": price_avg,
            "amount": total_amount_btc,
            "fee_usdt": fee_usdt,
            "ts_ms": ts_ms,
            "side": fills[0]["side"],
            "reduce_only": reduce_only,
        }
    return by_order


def _diagnose_match_failure(db_trade: dict, by_order: dict[str, dict]) -> str:
    """I-BLE004: 매칭 실패 trade 의 by_order 후보 검색 — 진단 로그 강화.

    어느 조건이 안 맞았는지 분류:
    - entry 후보 (ts ±60s + side 일치, size 무관): size mismatch 영역 파악용
    - exit reduce_only 후보 (size ±0.005 BTC + ts > db_ts + side 일치): entry 누락 case 의
      exit 단독 매칭 가능성 (I-BLE003 패턴: fetch_my_trades 가 entry 만 누락하고 exit 는 정상)

    출력 예 (Trade 1 case, I-BLE002 fix 검증 시점):
        entry buy 후보 (ts ±60s, size 무관): 1건 [250843897389056 size=0.0235]
        exit sell reduce_only 후보 (size ±0.005 BTC): 1건 [205647822848 size=0.0750]
    """
    side_str = db_trade["side"]
    entry_side = "buy" if side_str == "long" else "sell"
    exit_side = "sell" if side_str == "long" else "buy"
    db_size = float(db_trade["size"])
    db_ts_ms = _parse_iso_ms(db_trade["timestamp"])

    # entry 후보 (ts ±60s + side 일치 + reduce_only=False, size 무관)
    entry_ts_cands = [
        v for v in by_order.values()
        if abs(v["ts_ms"] - db_ts_ms) <= TIME_MARGIN_MS
        and v["side"] == entry_side
        and not v["reduce_only"]
    ]
    # exit reduce_only 후보 (ts > db_ts + side 일치 + reduce_only=True + size 일치)
    exit_size_cands = [
        v for v in by_order.values()
        if v["ts_ms"] > db_ts_ms
        and v["side"] == exit_side
        and v["reduce_only"]
        and abs(v["amount"] - db_size) <= SIZE_TOLERANCE_BTC
    ]

    def _fmt(v: dict) -> str:
        return f"{v['order_id'][-12:]} size={v['amount']:.4f}"

    entry_str = (
        f"entry {entry_side} 후보 (ts ±60s, size 무관): {len(entry_ts_cands)}건"
        + (f" [{', '.join(_fmt(v) for v in entry_ts_cands[:3])}]" if entry_ts_cands else "")
    )
    exit_str = (
        f"exit {exit_side} reduce_only 후보 (size ±0.005 BTC, ts > db_ts): "
        f"{len(exit_size_cands)}건"
        + (f" [{', '.join(_fmt(v) for v in exit_size_cands[:3])}]" if exit_size_cands else "")
    )
    return f"{entry_str}\n   {exit_str}"


def _match_trade(db_trade: dict, by_order: dict[str, dict]) -> tuple[dict | None, dict | None]:
    """id 우선 매칭 → fallback 시간/방향/size + reduceOnly."""
    side_str = db_trade["side"]  # 'long' or 'short'
    entry_side = "buy" if side_str == "long" else "sell"
    exit_side = "sell" if side_str == "long" else "buy"
    db_size = float(db_trade["size"])
    db_ts_ms = _parse_iso_ms(db_trade["timestamp"])

    # 4-a. id 우선 매칭
    entry: dict | None = None
    exit_: dict | None = None
    if db_trade.get("entry_order_id"):
        entry = by_order.get(db_trade["entry_order_id"])
    if db_trade.get("exit_order_id"):
        exit_ = by_order.get(db_trade["exit_order_id"])
    if entry and exit_:
        return entry, exit_

    # 4-b. fallback 시간/방향/size + reduceOnly
    if entry is None:
        entry_candidates = [
            v for v in by_order.values()
            if abs(v["ts_ms"] - db_ts_ms) <= TIME_MARGIN_MS
            and v["side"] == entry_side
            and not v["reduce_only"]
            and abs(v["amount"] - db_size) <= SIZE_TOLERANCE_BTC
        ]
        if entry_candidates:
            entry = min(entry_candidates, key=lambda v: abs(v["ts_ms"] - db_ts_ms))

    if exit_ is None and entry is not None:
        # Exit 는 entry 후 + reduceOnly=True + 반대 방향 + size 일치
        exit_candidates = [
            v for v in by_order.values()
            if v["ts_ms"] > entry["ts_ms"]
            and v["side"] == exit_side
            and v["reduce_only"]
            and abs(v["amount"] - db_size) <= SIZE_TOLERANCE_BTC
        ]
        if exit_candidates:
            exit_ = min(exit_candidates, key=lambda v: v["ts_ms"])

    return entry, exit_


def _recalc_pnl(
    entry: dict, exit_: dict, side_str: str, funding_fee: float,
) -> float:
    """OKX 실값 기반 PnL 재계산. fee_model.calc_pnl 흐름과 일관.

    net_pnl = (exit - entry) × size × side_sign - entry_fee - exit_fee - funding_fee
    """
    side_sign = 1 if side_str == "long" else -1
    gross = (exit_["price_avg"] - entry["price_avg"]) * entry["amount"] * side_sign
    return gross - entry["fee_usdt"] - exit_["fee_usdt"] - float(funding_fee)
