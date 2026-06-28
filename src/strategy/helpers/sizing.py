"""사이징 공식 (opt-in). 모델의 compute_position_size 에서 골라 쓴다."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def risk_based_size(
    entry_price: float,
    stop_price: float,
    balance: float,
    *,
    risk_per_trade_pct: float,
    max_leverage: float,
    max_position_size: float = float("inf"),
    volatility_factor: float = 1.0,
) -> float:
    """risk% · SL거리 기반 사이징.

    risk_amount = balance × risk_per_trade_pct
    raw_size    = risk_amount / |entry - stop|
    leverage / max_position_size 클램프, volatility_factor(>1 이면 축소) 적용.

    SL 거리에 의존하므로 stop_price 가 없으면(standing SL 미설정) 이 공식은 못 쓴다 —
    모델이 다른 방식을 택해야 한다.
    """
    if entry_price <= 0 or balance <= 0:
        return 0.0
    price_risk = abs(entry_price - stop_price)
    if price_risk <= 0:
        logger.warning("risk_based_size: invalid stop distance, returning 0")
        return 0.0
    raw_size = (balance * risk_per_trade_pct) / price_risk
    if volatility_factor > 0:
        raw_size *= min(1.0, 1.0 / volatility_factor)
    max_size_by_leverage = (balance * max_leverage) / entry_price
    return max(0.0, min(raw_size, max_size_by_leverage, max_position_size))
