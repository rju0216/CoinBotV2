"""FeeModel.calc_pnl funding 부호 검증.

funding 의미: holder net 영향 (양수=수익, 음수=비용).
net = gross - fees + funding
"""

from __future__ import annotations

import pytest

from src.accounting.fee_model import FeeModel
from src.core.enums import PositionSide


def _make_fm():
    return FeeModel(taker_fee_pct=0.0005, slippage_pct=0.0, funding_enabled=True)


class TestCalcPnlFundingSign:
    """calc_pnl funding 부호 검증."""

    def test_long_positive_funding_added_to_net(self):
        """LONG + 수익 funding (양수) → net 가산."""
        fm = _make_fm()
        result = fm.calc_pnl(
            side=PositionSide.LONG,
            entry_price=80000.0,
            exit_price=81000.0,
            size=0.1,
            fees=4.0,    # entry_fee + exit_fee 합
            funding=2.0,  # 양수 (수익)
        )
        # gross = (81000 - 80000) × 0.1 = 100
        assert result["gross_pnl"] == pytest.approx(100.0)
        # net = gross - fees + funding = 100 - 4 + 2 = 98
        assert result["net_pnl"] == pytest.approx(98.0)
        # pnl_pct = 98 / (80000 × 0.1) × 100 = 1.225%
        assert result["pnl_pct"] == pytest.approx(1.225)

    def test_long_negative_funding_subtracted_from_net(self):
        """LONG + 비용 funding (음수) → net 차감 (음수 가산으로 차감 효과)."""
        fm = _make_fm()
        result = fm.calc_pnl(
            side=PositionSide.LONG,
            entry_price=80000.0,
            exit_price=81000.0,
            size=0.1,
            fees=4.0,
            funding=-1.5,   # 음수 (비용)
        )
        # net = gross - fees + funding = 100 - 4 + (-1.5) = 94.5
        assert result["net_pnl"] == pytest.approx(94.5)

    def test_short_positive_funding_added_to_net(self):
        """SHORT + 수익 funding (양수) → net 가산. 실거래 케이스 재현."""
        fm = _make_fm()
        result = fm.calc_pnl(
            side=PositionSide.SHORT,
            entry_price=73446.6,
            exit_price=72637.5,
            size=0.127,
            fees=9.28,
            funding=0.84,    # 양수 (수익) — 실 OKX 거래 값
        )
        # gross = (73446.6 - 72637.5) × 0.127 = 102.7557
        assert result["gross_pnl"] == pytest.approx(102.7557, abs=0.01)
        # net = 102.76 - 9.28 + 0.84 = 94.32 — 실 OKX 표기 일치
        assert result["net_pnl"] == pytest.approx(94.32, abs=0.01)

    def test_zero_funding_unchanged(self):
        """funding=0 → net = gross - fees."""
        fm = _make_fm()
        result = fm.calc_pnl(
            side=PositionSide.LONG,
            entry_price=80000.0,
            exit_price=81000.0,
            size=0.1,
            fees=4.0,
            funding=0.0,
        )
        assert result["net_pnl"] == pytest.approx(96.0)   # 100 - 4

    def test_short_net_loss_with_negative_funding(self):
        """SHORT 손실 + 비용 funding → 더 큰 손실."""
        fm = _make_fm()
        result = fm.calc_pnl(
            side=PositionSide.SHORT,
            entry_price=80000.0,
            exit_price=80500.0,   # 상승 → SHORT 손실
            size=0.1,
            fees=4.0,
            funding=-1.0,   # 비용
        )
        # gross = (80000 - 80500) × 0.1 = -50
        assert result["gross_pnl"] == pytest.approx(-50.0)
        # net = -50 - 4 + (-1) = -55
        assert result["net_pnl"] == pytest.approx(-55.0)
