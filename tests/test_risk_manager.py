"""RiskManager 단위 테스트 (단계 4).

_legacy/tests의 사이징·검증 테스트를 신규 시그니처에 맞춰 이전. 동방향 차단,
SL/TP 산정, owner 분기 테스트는 책임이 옮겨갔으므로 제외.
"""

import pytest

from src.risk.manager import RiskManager


def _make_config(**overrides):
    base = {
        "risk": {
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.15,
            "max_position_size_btc": 1.0,
            "max_concurrent_positions": 1,
        }
    }
    base["risk"].update(overrides)
    return base


SIZING_KW = {"risk_per_trade_pct": 0.015, "max_leverage": 10}


class TestPositionSizing:
    def test_basic(self):
        rm = RiskManager(_make_config())
        # $10,000 balance, entry $67,000, stop $66,370 (risk $630)
        size = rm.calculate_position_size(67000, 66370, 10000, **SIZING_KW)
        # risk_amount = 10000 * 0.015 = $150 → raw = 150/630 ≈ 0.238 BTC
        assert 0.20 < size < 0.30

    def test_respects_max_leverage(self):
        rm = RiskManager(_make_config())
        # 매우 좁은 stop → 큰 raw size, leverage로 제한되어야 함
        size = rm.calculate_position_size(67000, 66990, 10000, **SIZING_KW)
        max_by_leverage = (10000 * 10) / 67000  # ≈ 1.49 BTC
        assert size <= max_by_leverage + 1e-3

    def test_respects_max_position_size(self):
        rm = RiskManager(_make_config())
        size = rm.calculate_position_size(67000, 66990, 1_000_000, **SIZING_KW)
        assert size <= 1.0  # max_position_size_btc

    def test_zero_risk_returns_zero(self):
        rm = RiskManager(_make_config())
        size = rm.calculate_position_size(67000, 67000, 10000, **SIZING_KW)
        assert size == 0.0

    def test_zero_balance_returns_zero(self):
        rm = RiskManager(_make_config())
        size = rm.calculate_position_size(67000, 66000, 0, **SIZING_KW)
        assert size == 0.0

    def test_strategy_specific_params(self):
        """전략이 자기 risk_per_trade_pct를 넘기면 그 값으로 사이징."""
        rm = RiskManager(_make_config())
        size_low = rm.calculate_position_size(
            67000, 66370, 10000, risk_per_trade_pct=0.005, max_leverage=5
        )
        size_high = rm.calculate_position_size(
            67000, 66370, 10000, risk_per_trade_pct=0.030, max_leverage=5
        )
        assert size_high > size_low
        assert size_high < 6 * size_low + 1e-6  # ratio 6배 미만 (leverage cap)


# ─── BL-2-1: EventBus 통합 ───


class _MockEventBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def publish(self, event_type, data):
        self.events.append((event_type, data))


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_drawdown_lock_publishes_event(self):
        import asyncio
        rm = RiskManager(_make_config())
        bus = _MockEventBus()
        rm.attach_event_bus(bus)
        rm.set_initial_balance(10000)
        rm.peak_equity = 10000
        # balance $8400 → 16% drawdown > 15% limit
        assert rm.validate_order(8400, current_position_count=0) is False
        assert rm.is_drawdown_locked is True
        # publish는 fire-and-forget create_task — 잠시 대기
        await asyncio.sleep(0.01)
        assert any(e[0] == "drawdown_locked" for e in bus.events)
        evt = next(e[1] for e in bus.events if e[0] == "drawdown_locked")
        assert "drawdown_pct" in evt
        assert "balance" in evt

    @pytest.mark.asyncio
    async def test_daily_loss_publishes_event_once(self):
        import asyncio
        rm = RiskManager(_make_config())
        bus = _MockEventBus()
        rm.attach_event_bus(bus)
        rm.set_initial_balance(10000)
        rm.daily_pnl = -600  # -6% > 5% limit
        # 3회 호출 — publish는 1회만
        for _ in range(3):
            assert rm.validate_order(10000, current_position_count=0) is False
        await asyncio.sleep(0.01)
        events = [e for e in bus.events if e[0] == "daily_loss_locked"]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_reset_daily_pnl_allows_new_publish(self):
        import asyncio
        rm = RiskManager(_make_config())
        bus = _MockEventBus()
        rm.attach_event_bus(bus)
        rm.set_initial_balance(10000)
        rm.daily_pnl = -600
        rm.validate_order(10000, current_position_count=0)
        await asyncio.sleep(0.01)
        rm.reset_daily_pnl()
        # 다시 daily loss 트리거 → 두 번째 publish
        rm.daily_pnl = -600
        rm.validate_order(10000, current_position_count=0)
        await asyncio.sleep(0.01)
        events = [e for e in bus.events if e[0] == "daily_loss_locked"]
        assert len(events) == 2

    def test_no_event_bus_does_not_crash(self):
        # event_bus 미연결 시에도 정상 동작 (logger만)
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm.peak_equity = 10000
        assert rm.validate_order(8400, current_position_count=0) is False
        assert rm.is_drawdown_locked is True


class TestVolatilityFactor:
    """BP-2-2 동적 사이징 (사안 J 가: 축소만)."""

    def test_default_factor_unchanged(self):
        rm = RiskManager(_make_config())
        size_default = rm.calculate_position_size(67000, 66370, 10000, **SIZING_KW)
        size_factor1 = rm.calculate_position_size(
            67000, 66370, 10000, volatility_factor=1.0, **SIZING_KW
        )
        assert size_default == size_factor1

    def test_factor_above_one_reduces_size(self):
        rm = RiskManager(_make_config())
        base = rm.calculate_position_size(67000, 66370, 10000, **SIZING_KW)
        half = rm.calculate_position_size(
            67000, 66370, 10000, volatility_factor=2.0, **SIZING_KW
        )
        assert half == pytest.approx(base / 2.0, rel=1e-6)

    def test_factor_below_one_no_increase(self):
        """축소만 (사안 J 가): factor<1 (잔잔)일 때 size 증가 없음."""
        rm = RiskManager(_make_config())
        base = rm.calculate_position_size(67000, 66370, 10000, **SIZING_KW)
        low_vol = rm.calculate_position_size(
            67000, 66370, 10000, volatility_factor=0.5, **SIZING_KW
        )
        assert low_vol == base

    def test_zero_factor_fallback(self):
        """target=0 등 잘못된 값 → factor=0 → fallback (변화 없음)."""
        rm = RiskManager(_make_config())
        base = rm.calculate_position_size(67000, 66370, 10000, **SIZING_KW)
        zero = rm.calculate_position_size(
            67000, 66370, 10000, volatility_factor=0.0, **SIZING_KW
        )
        assert zero == base

    def test_extreme_factor_clamped_by_zero_floor(self):
        """factor 매우 큼 → size 매우 작음. max(0)으로 음수 방지."""
        rm = RiskManager(_make_config())
        size = rm.calculate_position_size(
            67000, 66370, 10000, volatility_factor=1000.0, **SIZING_KW
        )
        assert size >= 0.0


class TestValidateOrder:
    def test_passes_normally(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        assert rm.validate_order(10000, current_position_count=0) is True

    def test_blocks_when_slot_full(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        assert rm.validate_order(10000, current_position_count=1) is False

    def test_blocks_daily_loss(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm.daily_pnl = -600  # -6% > 5% limit
        assert rm.validate_order(10000, current_position_count=0) is False

    def test_blocks_drawdown_and_locks(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm.peak_equity = 10000
        # balance $8400 → 16% drawdown > 15% limit
        assert rm.validate_order(8400, current_position_count=0) is False
        assert rm.is_drawdown_locked is True

    def test_blocks_when_dd_locked(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm._dd_locked = True
        assert rm.validate_order(10000, current_position_count=0) is False


class TestDrawdownUnlock:
    def test_unlock_sets_baseline(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm.peak_equity = 10000
        rm._dd_locked = True
        rm.unlock_drawdown(8000)
        assert rm.is_drawdown_locked is False
        assert rm.unlock_baseline == 8000
        # peak_equity 보존
        assert rm.peak_equity == 10000

    def test_baseline_clears_on_new_peak(self):
        rm = RiskManager(_make_config())
        rm.set_initial_balance(10000)
        rm.peak_equity = 10000
        rm.unlock_drawdown(8000)
        # 신고가 도달 → unlock_baseline 자동 폐기
        rm.update_equity(11000)
        assert rm.unlock_baseline is None
        assert rm.peak_equity == 11000


class TestPnLTracking:
    def test_add_and_reset(self):
        rm = RiskManager(_make_config())
        rm.add_pnl(100)
        rm.add_pnl(-50)
        assert rm.daily_pnl == 50
        rm.reset_daily_pnl()
        assert rm.daily_pnl == 0.0
