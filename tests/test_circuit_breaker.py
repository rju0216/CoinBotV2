"""LiveExecutor CircuitBreaker 단위 테스트 (BL-2-1 Step 2)."""

from __future__ import annotations

import ccxt.async_support as ccxt
import pytest

from src.execution.live_executor import (
    CircuitBreaker,
    CircuitBreakerOpen,
    _retry_api,
)


class TestCircuitBreakerState:
    def test_initial_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert not cb.is_open
        assert cb.consecutive_failures == 0

    def test_record_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.consecutive_failures == 2
        cb.record_success()
        assert cb.consecutive_failures == 0
        assert not cb.is_open

    def test_record_failure_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.record_failure() is False  # 1
        assert cb.record_failure() is False  # 2
        assert cb.record_failure() is True   # 3 → OPEN
        assert cb.is_open

    def test_check_or_raise_when_open(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()  # OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.check_or_raise()

    def test_check_or_raise_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.check_or_raise()  # 예외 없이 통과

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert not cb.is_open
        assert cb.consecutive_failures == 0


class TestRetryApiWithCircuitBreaker:
    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()  # 2/3
        async def ok():
            return "ok"
        result = await _retry_api(ok, retries=1, circuit_breaker=cb)
        assert result == "ok"
        assert cb.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_open_circuit_raises_immediately(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()  # OPEN
        async def ok():
            return "ok"
        with pytest.raises(CircuitBreakerOpen):
            await _retry_api(ok, retries=1, circuit_breaker=cb)

    @pytest.mark.asyncio
    async def test_consecutive_failures_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        async def fail():
            raise ccxt.NetworkError("simulated")

        # 3번 호출 (각각 retry 1회로 단순화) → consecutive_failures 3 → OPEN
        for _ in range(2):
            with pytest.raises(ccxt.NetworkError):
                await _retry_api(fail, retries=1, delay=0.001, circuit_breaker=cb)
        # 3번째 호출 시 카운터 도달 → CircuitBreakerOpen raise
        with pytest.raises(CircuitBreakerOpen):
            await _retry_api(fail, retries=1, delay=0.001, circuit_breaker=cb)
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_retry_api_without_breaker_works_as_before(self):
        async def ok():
            return 42
        result = await _retry_api(ok, retries=1)
        assert result == 42

    @pytest.mark.asyncio
    async def test_non_ccxt_exception_propagates(self):
        cb = CircuitBreaker(failure_threshold=3)
        async def boom():
            raise ValueError("not a network error")
        with pytest.raises(ValueError):
            await _retry_api(boom, retries=1, circuit_breaker=cb)
        # ccxt 예외 아니라 카운터 증가 안 함
        assert cb.consecutive_failures == 0


# ─── I-BL008: LiveExecutor._call이 _retry_api를 호출하는지 검증 ───


class TestLiveExecutorCallDelegation:
    """I-BL008 fix 검증: _call → _retry_api (자기 재귀 아님)."""

    @pytest.mark.asyncio
    async def test_call_delegates_to_retry_api(self, monkeypatch):
        """_call이 _retry_api 모듈 함수를 호출 (자기 자신 재귀가 아님)."""
        from src.execution import live_executor as live_module
        from src.execution.live_executor import LiveExecutor

        # _retry_api mock — 호출 인자 캡처
        captured: dict = {}

        async def mock_retry_api(func, *args, circuit_breaker=None, **kwargs):
            captured["func"] = func
            captured["args"] = args
            captured["circuit_breaker"] = circuit_breaker
            captured["kwargs"] = kwargs
            return "mocked_result"

        monkeypatch.setattr(live_module, "_retry_api", mock_retry_api)

        # LiveExecutor 인스턴스 생성 (ccxt 실 호출 없이)
        config = {
            "exchange": {
                "symbol": "BTC/USDT:USDT",
                "api_key": "test",
                "secret": "test",
                "passphrase": "test",
                "leverage": 5,
                "sandbox": True,
            },
            "risk": {"circuit_breaker": {"enabled": True, "failure_threshold": 5}},
        }
        executor = LiveExecutor(config)

        async def dummy_func(*a, **k):
            return "should_not_be_called"

        result = await executor._call(dummy_func, "arg1", kw1="value1")

        assert result == "mocked_result"
        assert captured["func"] is dummy_func
        assert captured["args"] == ("arg1",)
        assert captured["circuit_breaker"] is executor.circuit_breaker
        assert captured["kwargs"] == {"kw1": "value1"}

        await executor.close()

    @pytest.mark.asyncio
    async def test_close_position_skipped_when_exchange_already_closed(self, monkeypatch):
        """I-BL010: 거래소가 이미 청산했으면 redundant 주문 skip."""
        from src.core.enums import OrderType, PositionSide
        from src.execution.live_executor import LiveExecutor

        config = {
            "exchange": {
                "symbol": "BTC/USDT:USDT",
                "api_key": "test", "secret": "test", "passphrase": "test",
                "sandbox": True,
            },
            "risk": {"circuit_breaker": {"enabled": False}},
        }
        executor = LiveExecutor(config)

        # get_position이 None 반환 (거래소 ∅)
        async def mock_get_position():
            return None
        executor.get_position = mock_get_position

        # create_order는 호출되지 않아야 함 — 호출되면 테스트 실패
        async def mock_create_order(*args, **kwargs):
            raise AssertionError("create_order called despite empty position")
        executor.exchange.create_order = mock_create_order

        result = await executor.close_position(
            PositionSide.LONG, 0.075,
            order_type=OrderType.MARKET,
        )
        assert result["info"]["already_closed"] is True
        assert result["filled"] == 0
        await executor.close()

    @pytest.mark.asyncio
    async def test_close_position_proceeds_when_position_exists(self, monkeypatch):
        """I-BL010: 거래소 포지션 있으면 정상 close 흐름 진행."""
        from src.core.enums import OrderType, PositionSide
        from src.execution import live_executor as live_module
        from src.execution.live_executor import LiveExecutor

        config = {
            "exchange": {
                "symbol": "BTC/USDT:USDT",
                "api_key": "test", "secret": "test", "passphrase": "test",
                "sandbox": True,
            },
            "risk": {"circuit_breaker": {"enabled": False}},
        }
        executor = LiveExecutor(config)
        executor.contract_size = 0.01

        # get_position이 포지션 반환
        async def mock_get_position():
            return {"side": PositionSide.LONG, "size": 0.075, "entry_price": 67000.0}
        executor.get_position = mock_get_position

        # _retry_api mock — create_order 호출 캡처
        captured = {}
        async def mock_retry(func, *args, circuit_breaker=None, **kwargs):
            captured["called"] = True
            captured["args"] = args
            return {"id": "test_order", "filled": 7.5}
        monkeypatch.setattr(live_module, "_retry_api", mock_retry)

        result = await executor.close_position(
            PositionSide.LONG, 0.075, order_type=OrderType.MARKET,
        )
        assert captured["called"] is True
        assert result["id"] == "test_order"
        await executor.close()

    @pytest.mark.asyncio
    async def test_call_circuit_breaker_disabled(self, monkeypatch):
        """circuit_breaker=False config 시 None이 _retry_api에 전달됨."""
        from src.execution import live_executor as live_module
        from src.execution.live_executor import LiveExecutor

        captured: dict = {}

        async def mock_retry_api(func, *args, circuit_breaker=None, **kwargs):
            captured["circuit_breaker"] = circuit_breaker
            return None

        monkeypatch.setattr(live_module, "_retry_api", mock_retry_api)

        config = {
            "exchange": {
                "symbol": "BTC/USDT:USDT",
                "api_key": "test",
                "secret": "test",
                "passphrase": "test",
                "sandbox": True,
            },
            "risk": {"circuit_breaker": {"enabled": False}},
        }
        executor = LiveExecutor(config)
        assert executor.circuit_breaker is None

        async def dummy(): return None
        await executor._call(dummy)

        assert captured["circuit_breaker"] is None
        await executor.close()
