"""라이브 모드 recovery 영역 단위 테스트 (I-BL011/I-BL012/I-BL013).

- I-BL012: engine_base.close_position의 broker.close_position 실패 시 강건성
- I-BL013: _restore_state case 2의 fetch_actual_exit 정확성
- I-BL011: SL/TP conditional order 검증 + 재등록
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import ExitReason, PositionSide, PositionStatus
from src.core.types import Position


# ─── I-BL012: close_position 강건성 ───


class TestEngineCloseRobustness:
    """engine_base.close_position이 broker.close_position 실패 시에도 시스템 정리."""

    @pytest.mark.asyncio
    async def test_close_proceeds_when_broker_fails_but_exchange_empty(self):
        """broker.close_position exception + 거래소 ∅ → 시스템 상태 정리 진행."""
        from src.core.engine_base import AbstractEngine
        from src.accounting.fee_model import FeeModel

        # 최소 mock engine
        class _MockEngine:
            close_position = AbstractEngine.close_position

            def __init__(self):
                self._position = Position(
                    side=PositionSide.LONG, size=0.075, entry_price=82000.0,
                    entry_time=datetime(2026, 5, 6, 21, 15, tzinfo=timezone.utc),
                    strategy_name="ensemble", stop_loss=81700.0, take_profit=83000.0,
                    trade_id=1, status=PositionStatus.OPEN,
                )
                self.broker = MagicMock()
                self.broker.cancel_all_orders = AsyncMock()
                # close_position이 ExchangeError 같은 exception 발생
                self.broker.close_position = AsyncMock(
                    side_effect=Exception("simulated 51169 reject")
                )
                # get_position이 None (이미 청산)
                self.broker.get_position = AsyncMock(return_value=None)
                self.broker.get_balance = AsyncMock(return_value=3370.0)
                self.fee_model = FeeModel(taker_fee_pct=0.0005, slippage_pct=0.0)
                self.risk_manager = MagicMock()
                self.risk_manager.add_pnl = MagicMock()
                self.risk_manager.update_equity = MagicMock()
                self.strategy_by_name = {}
                self.event_bus = MagicMock()
                self.event_bus.publish = AsyncMock()
                self._record_trade_close = AsyncMock()
                self._latest_orderbook = None

        engine = _MockEngine()
        # 정상 진행해야 함 (exception propagate X)
        await engine.close_position(
            exit_price=81707.2, reason=ExitReason.SL_HIT, funding_fee=0.0,
        )

        # 시스템 상태 정리 확인
        assert engine._position is None  # 정리 완료
        assert engine._record_trade_close.called  # DB close 호출
        assert engine.event_bus.publish.called  # 이벤트 publish 호출
        assert engine.risk_manager.add_pnl.called  # PnL 누적

    @pytest.mark.asyncio
    async def test_close_propagates_when_exchange_still_has_position(self):
        """broker.close_position exception + 거래소 O → 진짜 청산 실패. exception propagate."""
        from src.core.engine_base import AbstractEngine
        from src.accounting.fee_model import FeeModel

        class _MockEngine:
            close_position = AbstractEngine.close_position

            def __init__(self):
                self._position = Position(
                    side=PositionSide.LONG, size=0.075, entry_price=82000.0,
                    entry_time=datetime(2026, 5, 6, tzinfo=timezone.utc),
                    strategy_name="ensemble", stop_loss=81700.0, take_profit=83000.0,
                    trade_id=1, status=PositionStatus.OPEN,
                )
                self.broker = MagicMock()
                self.broker.cancel_all_orders = AsyncMock()
                self.broker.close_position = AsyncMock(
                    side_effect=Exception("real network failure")
                )
                # 거래소에 포지션 살아있음
                self.broker.get_position = AsyncMock(return_value={
                    "side": PositionSide.LONG, "size": 0.075, "entry_price": 82000.0,
                })
                self.broker.get_balance = AsyncMock(return_value=3370.0)
                self.fee_model = FeeModel(taker_fee_pct=0.0005, slippage_pct=0.0)
                self.risk_manager = MagicMock()
                self.strategy_by_name = {}
                self.event_bus = MagicMock()
                self.event_bus.publish = AsyncMock()
                self._record_trade_close = AsyncMock()
                self._latest_orderbook = None

        engine = _MockEngine()

        with pytest.raises(Exception, match="real network failure"):
            await engine.close_position(
                exit_price=81707.2, reason=ExitReason.SL_HIT,
            )
        # 진짜 실패 — 시스템 상태 그대로 (사용자 manual 개입 영역)
        assert engine._position is not None
        assert not engine._record_trade_close.called


# ─── I-BL013: _restore_state case 2 fetch_actual_exit ───


class TestFetchActualExit:
    """_fetch_actual_exit이 거래소 fetch_closed_orders에서 정확한 청산 정보 추출."""

    @pytest.mark.asyncio
    async def test_fetch_returns_actual_exit_data(self):
        """fetch_closed_orders에서 reduceOnly + 반대 방향 order 찾아 exit_price/pnl 반환."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = True
                executor = MagicMock()
                executor.exchange = MagicMock()
                # OKX 청산 order mock — sell side, reduceOnly, avgPx
                executor.exchange.fetch_closed_orders = AsyncMock(return_value=[
                    {
                        "info": {"reduceOnly": "true"},
                        "side": "sell",
                        "average": 81707.297,
                        "status": "closed",
                    },
                ])
                self.broker.executor = executor
                self.config = {
                    "exchange": {"symbol": "BTC/USDT:USDT"},
                    "accounting": {"taker_fee_pct": 0.0005},
                }

        engine = _MockEngine()
        trade = {
            "id": 1, "side": "long", "size": 0.075,
            "entry_price": 82159.5, "stop_loss": 81710.28, "take_profit": 83057.95,
            "timestamp": "2026-05-06T21:15:01+00:00",
        }
        result = await engine._fetch_actual_exit(trade)
        assert result is not None
        # I-BL016: 4-튜플로 확장 (exit_price, pnl, reason, exit_ts_ms)
        exit_price, pnl, reason, _ = result
        assert exit_price == pytest.approx(81707.297)
        # gross = (81707.297 - 82159.5) × 0.075 = -33.915
        # entry_fee = 82159.5 × 0.075 × 0.0005 = 3.081
        # close_fee = 81707.297 × 0.075 × 0.0005 = 3.064
        # net = -33.915 - 3.081 - 3.064 ≈ -40.06
        assert pnl == pytest.approx(-40.06, abs=0.1)
        assert reason == ExitReason.SL_HIT.value  # exit < entry → SL

    @pytest.mark.asyncio
    async def test_fetch_skips_non_reduceonly_orders(self):
        """reduceOnly=false 인 order는 skip (진입 order 등)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = True
                executor = MagicMock()
                executor.exchange = MagicMock()
                # 진입 order(buy, reduceOnly=false) + 청산 order(sell, reduceOnly=true)
                executor.exchange.fetch_closed_orders = AsyncMock(return_value=[
                    {
                        "info": {"reduceOnly": "false"},
                        "side": "buy", "average": 82170.1, "status": "closed",
                    },
                    {
                        "info": {"reduceOnly": "true"},
                        "side": "sell", "average": 81707.297, "status": "closed",
                    },
                ])
                self.broker.executor = executor
                self.config = {
                    "exchange": {"symbol": "BTC/USDT:USDT"},
                    "accounting": {"taker_fee_pct": 0.0005},
                }

        engine = _MockEngine()
        result = await engine._fetch_actual_exit({
            "id": 1, "side": "long", "size": 0.075, "entry_price": 82159.5,
            "timestamp": "2026-05-06T21:15:01+00:00",
        })
        assert result is not None
        exit_price, _, _, _ = result
        # 진입 order(82170)이 아닌 reduceOnly=true 청산 order(81707) 선택
        assert exit_price == pytest.approx(81707.297)

    @pytest.mark.asyncio
    async def test_fetch_skips_same_side_orders(self):
        """반대 방향이 아닌 order는 skip (예: long 포지션의 buy reduceOnly는 hedge 모드 가능성)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = True
                executor = MagicMock()
                executor.exchange = MagicMock()
                # buy reduceOnly만 있고 sell이 없음 (long 청산이 아님)
                executor.exchange.fetch_closed_orders = AsyncMock(return_value=[
                    {
                        "info": {"reduceOnly": "true"},
                        "side": "buy", "average": 81700.0, "status": "closed",
                    },
                ])
                self.broker.executor = executor
                self.config = {
                    "exchange": {"symbol": "BTC/USDT:USDT"},
                    "accounting": {"taker_fee_pct": 0.0005},
                }

        engine = _MockEngine()
        result = await engine._fetch_actual_exit({
            "id": 1, "side": "long", "size": 0.075, "entry_price": 82159.5,
            "timestamp": "2026-05-06T21:15:01+00:00",
        })
        assert result is None  # 매칭되는 close order 없음

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_paper_mode(self):
        """paper 모드에서는 fetch 시도 안 함 (None 반환)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = False

        engine = _MockEngine()
        result = await engine._fetch_actual_exit({"id": 1})
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_api_failure(self):
        """fetch_closed_orders exception 시 None 반환 (caller가 fallback 처리)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = True
                executor = MagicMock()
                executor.exchange = MagicMock()
                executor.exchange.fetch_closed_orders = AsyncMock(
                    side_effect=Exception("API rate limit")
                )
                self.broker.executor = executor
                self.config = {"exchange": {"symbol": "BTC/USDT:USDT"}}

        engine = _MockEngine()
        result = await engine._fetch_actual_exit({
            "id": 1, "side": "long", "size": 0.075, "entry_price": 82000.0,
            "timestamp": "2026-05-06T21:15:01+00:00",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_short_position_tp_hit(self):
        """SHORT 포지션에서 exit < entry는 TP (profit), exit > entry는 SL (loss)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _fetch_actual_exit = CoreEngine._fetch_actual_exit

            def __init__(self):
                self.broker = MagicMock()
                self.broker.is_live = True
                executor = MagicMock()
                executor.exchange = MagicMock()
                # SHORT 청산 = buy reduceOnly. exit < entry → TP_HIT
                executor.exchange.fetch_closed_orders = AsyncMock(return_value=[
                    {
                        "info": {"reduceOnly": "true"},
                        "side": "buy", "average": 81000.0, "status": "closed",
                    },
                ])
                self.broker.executor = executor
                self.config = {
                    "exchange": {"symbol": "BTC/USDT:USDT"},
                    "accounting": {"taker_fee_pct": 0.0005},
                }

        engine = _MockEngine()
        result = await engine._fetch_actual_exit({
            "id": 1, "side": "short", "size": 0.075, "entry_price": 82000.0,
            "timestamp": "2026-05-06T21:15:01+00:00",
        })
        assert result is not None
        _, pnl, reason, _ = result
        assert reason == ExitReason.TP_HIT.value
        assert pnl > 0  # short + 가격 하락 → 수익


# ─── I-BL011: _verify_and_restore_sl_tp ───


class TestSyncUnexpectedClose:
    """I-BL015: 거래소가 모르게 청산한 포지션 동기화 (LONG/SHORT 모두)."""

    def _make_engine(self, side: PositionSide):
        """공통 mock engine 빌더."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _position_to_trade_dict = CoreEngine._position_to_trade_dict
            _sync_unexpected_close = CoreEngine._sync_unexpected_close

            def __init__(self):
                self._position = Position(
                    side=side,
                    size=0.075,
                    entry_price=82000.0,
                    entry_time=datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
                    strategy_name="ensemble",
                    stop_loss=82500.0 if side == PositionSide.SHORT else 81500.0,
                    take_profit=81000.0 if side == PositionSide.SHORT else 83000.0,
                    trade_id=1,
                )
                self.broker = MagicMock()
                self.broker.is_live = True
                self._fetch_actual_exit = AsyncMock()
                self._close_with_funding = AsyncMock()

        return _MockEngine()

    @pytest.mark.asyncio
    async def test_short_tp_spike_sync_long_path(self):
        """SHORT TP spike 청산 — fetch 성공 시 정확한 exit/reason 사용."""
        engine = self._make_engine(PositionSide.SHORT)
        # I-BL016: 4-튜플 (exit_price, pnl, reason, exit_ts_ms). 운영 중 _sync_unexpected_close
        # 는 timestamp 미사용이라 None 으로 mock
        engine._fetch_actual_exit.return_value = (
            81000.0, 60.39, ExitReason.TP_HIT.value, None,
        )
        await engine._sync_unexpected_close(
            last_known_price=81100.0,
            now=datetime(2026, 5, 8, 12, 15, tzinfo=timezone.utc),
        )
        # _close_with_funding 호출됨 (정상 close 흐름)
        assert engine._close_with_funding.called
        call_args = engine._close_with_funding.call_args
        assert call_args.args[0] == 81000.0  # exit_price (fetch에서)
        assert call_args.args[1] == ExitReason.TP_HIT  # reason

    @pytest.mark.asyncio
    async def test_long_sl_spike_sync(self):
        """LONG SL spike 청산 — fetch 성공 시 정확한 정보."""
        engine = self._make_engine(PositionSide.LONG)
        engine._fetch_actual_exit.return_value = (
            81500.0, -40.0, ExitReason.SL_HIT.value, None,
        )
        await engine._sync_unexpected_close(
            last_known_price=81600.0,
            now=datetime(2026, 5, 8, 12, 15, tzinfo=timezone.utc),
        )
        call_args = engine._close_with_funding.call_args
        assert call_args.args[0] == 81500.0
        assert call_args.args[1] == ExitReason.SL_HIT

    @pytest.mark.asyncio
    async def test_fetch_failure_uses_fallback(self, caplog):
        """fetch 실패 시 last_known_price + ENGINE_SHUTDOWN fallback."""
        import logging
        engine = self._make_engine(PositionSide.LONG)
        engine._fetch_actual_exit.return_value = None  # fetch 실패
        with caplog.at_level(logging.WARNING):
            await engine._sync_unexpected_close(
                last_known_price=81600.0,
                now=datetime(2026, 5, 8, 12, 15, tzinfo=timezone.utc),
            )
        call_args = engine._close_with_funding.call_args
        assert call_args.args[0] == 81600.0  # last_known_price
        assert call_args.args[1] == ExitReason.ENGINE_SHUTDOWN
        # WARNING 로그 출력
        assert any(
            "exchange trade history fetch failed" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_position_to_trade_dict(self):
        """_position_to_trade_dict이 _fetch_actual_exit 호환 dict 반환."""
        engine = self._make_engine(PositionSide.SHORT)
        d = engine._position_to_trade_dict()
        assert d["side"] == "short"
        assert d["size"] == 0.075
        assert d["entry_price"] == 82000.0
        assert d["stop_loss"] == 82500.0
        assert d["take_profit"] == 81000.0
        assert d["timestamp"] == "2026-05-08T12:00:00+00:00"


class TestVerifyAndRestoreSLTP:
    """SL/TP conditional order 생존 검증 + 누락 시 재등록."""

    @pytest.mark.asyncio
    async def test_both_alive_no_action(self):
        """SL/TP 둘 다 거래소에 살아있으면 재등록 안 함."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _verify_and_restore_sl_tp = CoreEngine._verify_and_restore_sl_tp

            def __init__(self):
                self._position = Position(
                    side=PositionSide.LONG, size=0.075, entry_price=82000.0,
                    entry_time=datetime(2026, 5, 6, tzinfo=timezone.utc),
                    strategy_name="ensemble", stop_loss=81700.0, take_profit=83000.0,
                )
                self.broker = MagicMock()
                self.broker.is_live = True
                self.broker.place_stop_loss = AsyncMock()
                self.broker.place_take_profit = AsyncMock()
                executor = MagicMock()
                executor.exchange = MagicMock()
                # 거래소에 SL/TP 둘 다 살아있음
                executor.exchange.fetch_open_orders = AsyncMock(return_value=[
                    {"info": {"slTriggerPx": "81700.0"}},
                    {"info": {"tpTriggerPx": "83000.0"}},
                ])
                self.broker.executor = executor
                self.config = {"exchange": {"symbol": "BTC/USDT:USDT"}}

        engine = _MockEngine()
        await engine._verify_and_restore_sl_tp()
        # 재등록 호출 안 됨
        assert not engine.broker.place_stop_loss.called
        assert not engine.broker.place_take_profit.called

    @pytest.mark.asyncio
    async def test_sl_missing_re_registers(self):
        """SL이 거래소에 없으면 재등록 호출."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _verify_and_restore_sl_tp = CoreEngine._verify_and_restore_sl_tp

            def __init__(self):
                self._position = Position(
                    side=PositionSide.LONG, size=0.075, entry_price=82000.0,
                    entry_time=datetime(2026, 5, 6, tzinfo=timezone.utc),
                    strategy_name="ensemble", stop_loss=81700.0, take_profit=83000.0,
                )
                self.broker = MagicMock()
                self.broker.is_live = True
                self.broker.place_stop_loss = AsyncMock()
                self.broker.place_take_profit = AsyncMock()
                executor = MagicMock()
                executor.exchange = MagicMock()
                # SL 누락 (TP만 있음)
                executor.exchange.fetch_open_orders = AsyncMock(return_value=[
                    {"info": {"tpTriggerPx": "83000.0"}},
                ])
                self.broker.executor = executor
                self.config = {"exchange": {"symbol": "BTC/USDT:USDT"}}

        engine = _MockEngine()
        await engine._verify_and_restore_sl_tp()
        assert engine.broker.place_stop_loss.called
        assert not engine.broker.place_take_profit.called

    @pytest.mark.asyncio
    async def test_fetch_failure_skip_silently(self):
        """fetch_open_orders 실패 시 재등록 시도 안 함 (거래소 정상 가정)."""
        from src.live.engine import CoreEngine

        class _MockEngine:
            _verify_and_restore_sl_tp = CoreEngine._verify_and_restore_sl_tp

            def __init__(self):
                self._position = Position(
                    side=PositionSide.LONG, size=0.075, entry_price=82000.0,
                    entry_time=datetime(2026, 5, 6, tzinfo=timezone.utc),
                    strategy_name="ensemble", stop_loss=81700.0, take_profit=83000.0,
                )
                self.broker = MagicMock()
                self.broker.is_live = True
                self.broker.place_stop_loss = AsyncMock()
                self.broker.place_take_profit = AsyncMock()
                executor = MagicMock()
                executor.exchange = MagicMock()
                executor.exchange.fetch_open_orders = AsyncMock(
                    side_effect=Exception("API error")
                )
                self.broker.executor = executor
                self.config = {"exchange": {"symbol": "BTC/USDT:USDT"}}

        engine = _MockEngine()
        # exception propagate 안 됨, 재등록도 안 함
        await engine._verify_and_restore_sl_tp()
        assert not engine.broker.place_stop_loss.called
        assert not engine.broker.place_take_profit.called
