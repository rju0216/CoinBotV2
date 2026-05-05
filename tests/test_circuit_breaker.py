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
