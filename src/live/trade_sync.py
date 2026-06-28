"""TradeSyncer — DB trades 를 OKX 실값으로 sync.

OKX positions-history 를 직접 사용한다 — pnl, funding_fee, entry/exit_price,
size, trading_fee, closed_at, pnl_pct 모두 OKX 실값. fetch_my_trades 로
entry/exit_order_id 를 추가 매칭한다.

알고리즘:
1. paper 가드 (broker.is_live=False → skip)
2. data_store.get_unsynced_trades() — closed AND synced_at IS NULL
3. earliest_ts - 60s 기준 두 API 호출:
   - _fetch_all_positions: positions-history (entry/exit/pnl/funding/closed_at)
   - _fetch_all_fills: fetch_my_trades (entry/exit_order_id 매칭용)
4. 각 미sync trade 매칭:
   - position 매칭: trade.closed_at vs uTime ±15분 + side
   - order_id 매칭: fills 에서 ts ±60s + size + side
5. DB UPDATE — OKX positions-history 직접 사용 (pnl_pct 는 net 기준 수식)
6. 매칭 실패 시 진단 로그 출력
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


SIZE_TOLERANCE_BTC = 0.005          # 1 contract (0.01 BTC) 의 half
TIME_MARGIN_MS = 60_000              # ±60초 (fills 매칭 + fetch since)
POSITION_TIME_MARGIN_MS = 15 * 60_000  # ±15분 (position 매칭: 외부 청산 시점 vs 라이브 인지 시점 차이 흡수)
PAGE_LIMIT = 100                     # OKX V5 한 page 한도
MAX_PAGES = 10                       # pagination 안전망


async def sync_all_unsynced(
    broker: Any,
    data_store: Any,
    symbol: str,
    accounting_config: dict | None = None,
) -> dict:
    """미sync trade 일괄 처리. CoreEngine._close_with_funding 직후 호출 (라이브 전용).

    Returns:
        {synced_count, failed_count, errors: list[str]}
    """
    # paper 가드
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

    # 2. earliest_ts 기준 OKX API 호출
    earliest_ts_ms = min(_parse_iso_ms(t["timestamp"]) for t in unsynced)
    since_ms = earliest_ts_ms - TIME_MARGIN_MS

    executor = broker.executor
    if executor is None or not hasattr(executor, "exchange"):
        return {"synced_count": 0, "failed_count": 0, "errors": ["broker.executor 없음"]}

    contract_size = float(getattr(executor, "contract_size", 1.0) or 1.0)
    if contract_size <= 0:
        contract_size = 1.0

    try:
        okx_positions = await _fetch_all_positions(executor.exchange, symbol, since_ms)
    except Exception as e:
        logger.warning("fetch_positions_history 실패 — sync skip: %s", e)
        return {
            "synced_count": 0,
            "failed_count": len(unsynced),
            "errors": [f"fetch_positions_history: {e}"],
        }

    try:
        okx_fills = await _fetch_all_fills(executor.exchange, symbol, since_ms)
    except Exception as e:
        logger.warning("fetch_my_trades 실패 — sync 영역 fills 매칭 제한: %s", e)
        okx_fills = []

    # order_id 별 fills 그룹화 (entry/exit_order_id 매칭용)
    by_order = _group_fills_by_order(okx_fills, contract_size)

    # 3. 각 미sync trade 매칭 + DB UPDATE
    results = {"synced_count": 0, "failed_count": 0, "errors": []}
    for db_trade in unsynced:
        try:
            position = _match_trade_to_position(db_trade, okx_positions)
            if position is None:
                # 매칭 실패 진단 로그
                diag = _diagnose_position_match_failure(db_trade, okx_positions, by_order)
                msg = (
                    f"trade {db_trade['id']} 매칭 실패 "
                    f"(side={db_trade['side']}, size={db_trade['size']:.6f}, "
                    f"closed_at={db_trade.get('closed_at') or db_trade['timestamp']})\n   {diag}"
                )
                logger.warning("Trade sync: %s", msg)
                results["failed_count"] += 1
                results["errors"].append(msg)
                continue

            # order_id 매칭 (DB 기존 값 유지 또는 fills 에서 추출)
            entry_oid, exit_oid = _resolve_order_ids(db_trade, position, by_order)

            # OKX positions-history 직접 사용
            openAvgPx = float(position["openAvgPx"])
            closeAvgPx = float(position["closeAvgPx"])
            size_btc = float(position["closeTotalPos"]) * contract_size
            trading_fee = abs(float(position.get("fee", 0) or 0))
            funding_fee = float(position.get("fundingFee", 0) or 0)
            pnl = float(position["realizedPnl"])
            notional = openAvgPx * size_btc
            pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
            closed_at_iso = _ms_to_iso(int(position["uTime"]))

            await data_store.update_synced_trade(
                trade_id=db_trade["id"],
                entry_price=openAvgPx,
                exit_price=closeAvgPx,
                size=size_btc,
                trading_fee=trading_fee,
                funding_fee=funding_fee,
                pnl=pnl,
                pnl_pct=pnl_pct,
                closed_at=closed_at_iso,
                entry_order_id=entry_oid,
                exit_order_id=exit_oid,
                synced_at=datetime.now(timezone.utc).isoformat(),
            )
            results["synced_count"] += 1
        except Exception as e:
            logger.warning("Trade sync trade %d exception: %s", db_trade.get("id"), e)
            results["failed_count"] += 1
            results["errors"].append(f"trade {db_trade.get('id')}: {e}")
    return results


# ---------- helper ----------

def _parse_iso_ms(iso_str: str) -> int:
    """ISO timestamp → UTC ms."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_iso(ms: int) -> str:
    """UTC ms → ISO timestamp."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


async def _fetch_all_positions(
    exchange: Any, symbol: str, since_ms: int,
) -> list[dict]:
    """ccxt fetch_positions_history 호출 + 각 항목의 raw info 추출.

    OKX positions-history 는 한 호출로 충분한 범위를 반환하므로
    pagination 은 안전망 수준(PAGE_LIMIT)만 둔다.
    """
    try:
        positions = await exchange.fetch_positions_history(
            symbols=[symbol], since=since_ms, limit=PAGE_LIMIT,
        )
    except Exception as e:
        logger.warning("fetch_positions_history exception: %s", e)
        raise
    # ccxt 항목의 info 는 raw OKX 응답 (필요 필드 포함)
    return [p.get("info", {}) for p in positions if p.get("info")]


async def _fetch_all_fills(
    exchange: Any, symbol: str, since_ms: int,
) -> list[dict]:
    """fetch_my_trades pagination — entry/exit_order_id 매칭용 fills 수집."""
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
            break
        all_fills.extend(new_fills)
        seen_ids.update(f["id"] for f in new_fills if f.get("id"))
        if len(fills) < PAGE_LIMIT:
            break
        last_ts = max(
            int(f["timestamp"]) for f in new_fills if f.get("timestamp")
        )
        if last_ts <= current_since:
            break
        current_since = last_ts
    else:
        logger.warning(
            "Trade sync: MAX_PAGES=%d 도달 — fills 누적 %d, 누락 가능.",
            MAX_PAGES, len(all_fills),
        )
    return all_fills


def _group_fills_by_order(
    okx_fills: list[dict], contract_size: float,
) -> dict[str, dict]:
    """order_id 별 fills 그룹화 — entry/exit_order_id 매칭에 사용.

    fillPnl 기반으로 reduce_only 판별 (entry order: fillPnl=0,
    exit order: fillPnl≠0). amount 는 contracts → BTC 로 변환한다.
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

        total_fill_pnl = sum(
            float((f.get("info") or {}).get("fillPnl", 0) or 0)
            for f in fills
        )
        reduce_only = abs(total_fill_pnl) > 1e-9
        ts_ms = min(int(f["timestamp"]) for f in fills if f.get("timestamp"))
        by_order[oid] = {
            "order_id": oid,
            "amount": total_amount_btc,
            "ts_ms": ts_ms,
            "side": fills[0]["side"],
            "reduce_only": reduce_only,
        }
    return by_order


def _match_trade_to_position(
    db_trade: dict, okx_positions: list[dict],
) -> dict | None:
    """trade.closed_at vs position.uTime ±15분 + side 로 매칭.

    ±15분: 외부 청산 (SL/TP trigger) 시점과 라이브 인지 시점 (봉 마감) 의
    차이를 흡수한다. 사용자 외 거래는 side 일치 제한 + size 검증으로
    자연스럽게 skip 된다.
    """
    closed_at_iso = db_trade.get("closed_at") or db_trade["timestamp"]
    db_closed_ms = _parse_iso_ms(closed_at_iso)
    db_side = db_trade["side"]  # 'long' or 'short'
    db_size = float(db_trade["size"])

    candidates = [
        p for p in okx_positions
        if p.get("uTime") and abs(int(p["uTime"]) - db_closed_ms) <= POSITION_TIME_MARGIN_MS
        and p.get("direction") == db_side
    ]
    if not candidates:
        return None
    # size 일치 검증 (사용자 외 거래 차단)
    contract_size_btc = 0.01  # OKX BTC-USDT-SWAP contract size
    size_filtered = []
    for p in candidates:
        try:
            okx_size_btc = float(p["closeTotalPos"]) * contract_size_btc
            if abs(okx_size_btc - db_size) <= SIZE_TOLERANCE_BTC:
                size_filtered.append(p)
        except (KeyError, ValueError, TypeError):
            continue
    if not size_filtered:
        return None
    return min(size_filtered, key=lambda p: abs(int(p["uTime"]) - db_closed_ms))


def _resolve_order_ids(
    db_trade: dict, position: dict, by_order: dict[str, dict],
) -> tuple[str | None, str | None]:
    """entry/exit_order_id 결정 — DB 기존 값 우선, 없으면 fills 에서 추출.

    fills 추출 기준: 시간/방향/size + reduce_only — entry order (reduce_only=False),
    exit order (reduce_only=True).
    """
    entry_oid = db_trade.get("entry_order_id")
    exit_oid = db_trade.get("exit_order_id")
    if entry_oid and exit_oid:
        return entry_oid, exit_oid

    side_str = db_trade["side"]
    entry_side = "buy" if side_str == "long" else "sell"
    exit_side = "sell" if side_str == "long" else "buy"
    db_size = float(db_trade["size"])
    db_ts_ms = _parse_iso_ms(db_trade["timestamp"])

    # entry 매칭
    if entry_oid is None:
        cands = [
            v for v in by_order.values()
            if abs(v["ts_ms"] - db_ts_ms) <= TIME_MARGIN_MS
            and v["side"] == entry_side
            and not v["reduce_only"]
            and abs(v["amount"] - db_size) <= SIZE_TOLERANCE_BTC
        ]
        if cands:
            entry_oid = min(cands, key=lambda v: abs(v["ts_ms"] - db_ts_ms))["order_id"]

    # exit 매칭
    if exit_oid is None:
        # closed_at 기반 매칭
        closed_at_iso = db_trade.get("closed_at") or db_trade["timestamp"]
        db_closed_ms = _parse_iso_ms(closed_at_iso)
        cands = [
            v for v in by_order.values()
            if abs(v["ts_ms"] - db_closed_ms) <= TIME_MARGIN_MS
            and v["side"] == exit_side
            and v["reduce_only"]
            and abs(v["amount"] - db_size) <= SIZE_TOLERANCE_BTC
        ]
        if cands:
            exit_oid = min(cands, key=lambda v: abs(v["ts_ms"] - db_closed_ms))["order_id"]

    return entry_oid, exit_oid


def _diagnose_position_match_failure(
    db_trade: dict, okx_positions: list[dict], by_order: dict[str, dict],
) -> str:
    """매칭 실패 시 진단 정보 문자열 생성."""
    db_side = db_trade["side"]
    closed_at_iso = db_trade.get("closed_at") or db_trade["timestamp"]
    db_closed_ms = _parse_iso_ms(closed_at_iso)

    # position 후보 (시간 ±15분, side 무관)
    pos_ts_cands = [
        p for p in okx_positions
        if p.get("uTime") and abs(int(p["uTime"]) - db_closed_ms) <= POSITION_TIME_MARGIN_MS
    ]
    # position 후보 (side 일치, 시간 무관)
    pos_side_cands = [
        p for p in okx_positions if p.get("direction") == db_side
    ]

    def _fmt_pos(p):
        ut = int(p.get("uTime", 0))
        return (
            f"posId=...{p.get('posId', '')[-12:]} "
            f"direction={p.get('direction', '')} "
            f"uTime={_ms_to_iso(ut)[:19] if ut else 'N/A'}"
        )

    parts = [
        f"position 후보 (uTime ±60s, side 무관): {len(pos_ts_cands)}건"
        + (f" [{', '.join(_fmt_pos(p) for p in pos_ts_cands[:3])}]" if pos_ts_cands else ""),
        f"position 후보 (side={db_side}, 시간 무관): {len(pos_side_cands)}건",
    ]
    return "\n   ".join(parts)
