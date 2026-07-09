import asyncio

import pytest

from local_ai_core.runtimes.errors import RequestTimeout
from local_ai_core.security.tool_call_timeout import with_timeout


class TestFastCallCompletesNormally:
    async def test_returns_the_function_result_when_it_finishes_in_time(self):
        async def fast_call() -> str:
            return "done"

        result = await with_timeout(fast_call, timeout_seconds=1.0)
        assert result == "done"


class TestSlowCallTimesOut:
    async def test_raises_request_timeout_when_the_call_hangs(self):
        async def slow_call() -> str:
            await asyncio.sleep(10)
            return "never gets here"

        with pytest.raises(RequestTimeout):
            await with_timeout(slow_call, timeout_seconds=0.01)

    async def test_the_underlying_asyncio_timeout_is_wrapped_as_the_cause(self):
        async def slow_call() -> str:
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(RequestTimeout) as exc_info:
            await with_timeout(slow_call, timeout_seconds=0.01)
        assert isinstance(exc_info.value.cause, asyncio.TimeoutError)


class TestInvalidTimeout:
    async def test_nonpositive_timeout_raises_value_error(self):
        async def fn() -> str:
            return "x"

        with pytest.raises(ValueError):
            await with_timeout(fn, timeout_seconds=0)
