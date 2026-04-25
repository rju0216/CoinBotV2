"""수수료·슬리피지·펀딩비 정산 모델.

라이브 엔진과 백테스트 엔진이 같은 클래스로 PnL을 정산하여 공식 일관성을 보장한다.
백테는 estimate_* 메서드로 사전 계산, 라이브는 record_actual_*로 실제 체결값을 기록.
"""

from __future__ import annotations

from typing import Any

from src.core.enums import PositionSide
from src.core.types import Fill, Position


class FeeModel:
    def __init__(
        self,
        taker_fee_pct: float = 0.0005,
        slippage_pct: float = 0.0,
        funding_enabled: bool = True,
    ) -> None:
        self.taker_fee_pct = float(taker_fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.funding_enabled = bool(funding_enabled)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeeModel":
        acc = config.get("accounting", {}) or {}
        return cls(
            taker_fee_pct=float(acc.get("taker_fee_pct", 0.0005)),
            slippage_pct=float(acc.get("slippage_pct", 0.0)),
            funding_enabled=bool(acc.get("funding_enabled", True)),
        )

    @property
    def per_side_rate(self) -> float:
        """체결 한쪽당 차감 비율 (수수료 + 슬리피지)."""
        return self.taker_fee_pct + self.slippage_pct

    def estimate_entry_fee(self, price: float, size: float) -> float:
        return price * size * self.per_side_rate

    def estimate_exit_fee(self, price: float, size: float) -> float:
        return price * size * self.per_side_rate

    def estimate_round_trip(
        self, entry_price: float, exit_price: float, size: float
    ) -> float:
        return self.estimate_entry_fee(entry_price, size) + self.estimate_exit_fee(
            exit_price, size
        )

    def record_actual_fee(self, fill: Fill) -> float:
        """라이브 체결 수수료 기록 — 거래소 응답값을 그대로 사용."""
        return float(fill.fee)

    def estimate_funding(self, position: Position, hours: float) -> float:
        """백테용 펀딩비 근사. 프로토타입은 0 반환.

        향후 확장: 평균 funding rate × hours / 8h × notional 등의 근사 모델로 교체 가능.
        """
        if not self.funding_enabled:
            return 0.0
        return 0.0

    def calc_pnl(
        self,
        side: PositionSide,
        entry_price: float,
        exit_price: float,
        size: float,
        fees: float = 0.0,
        funding: float = 0.0,
    ) -> dict[str, float]:
        if side == PositionSide.LONG:
            gross = (exit_price - entry_price) * size
        elif side == PositionSide.SHORT:
            gross = (entry_price - exit_price) * size
        else:
            gross = 0.0
        net = gross - fees - funding
        notional = entry_price * size
        pct = (net / notional * 100.0) if notional > 0 else 0.0
        return {
            "gross_pnl": gross,
            "fees": fees,
            "funding": funding,
            "net_pnl": net,
            "pnl_pct": pct,
        }
