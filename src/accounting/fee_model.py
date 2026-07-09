"""수수료·슬리피지·펀딩비 정산 모델.

PnL 은 `calc_pnl` 단일 공식으로 산출한다. 백테 엔진이 캔들 가격으로 체결을
시뮬하고 이 클래스로 왕복 수수료·순손익을 정산한다.
"""

from __future__ import annotations

from typing import Any

from src.core.enums import PositionSide
from src.core.types import Position


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
        """PnL 정산. funding 은 보유자 입장의 net 영향(부호 포함)으로 전달받는다.
        - 양수 funding: 수익 (가산) — 예: positive funding rate 구간의 short 보유
        - 음수 funding: 비용 (차감) — 예: positive funding rate 구간의 long 보유
        net = gross - fees + funding (funding 부호 그대로 가산)
        """
        if side == PositionSide.LONG:
            gross = (exit_price - entry_price) * size
        elif side == PositionSide.SHORT:
            gross = (entry_price - exit_price) * size
        else:
            gross = 0.0
        net = gross - fees + funding
        notional = entry_price * size
        pct = (net / notional * 100.0) if notional > 0 else 0.0
        return {
            "gross_pnl": gross,
            "fees": fees,
            "funding": funding,
            "net_pnl": net,
            "pnl_pct": pct,
        }
