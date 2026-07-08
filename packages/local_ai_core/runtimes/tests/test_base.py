import logging

import pytest

from local_ai_core.runtimes.base import (
    LoggingMetricsHook,
    NullMetricsHook,
    Timer,
    ensure_trace_id,
    with_retries,
)
from local_ai_core.runtimes.errors import (
    FeatureNotSupported,
    RequestTimeout,
    RuntimeUnavailable,
    SchemaValidationError,
)
from local_ai_core.runtimes.types import LLMRequest, LLMResponse


class TestEnsureTraceId:
    def test_fills_in_a_trace_id_when_missing(self):
        req = LLMRequest(model="m", prompt="p")
        assert req.trace_id is None
        updated = ensure_trace_id(req)
        assert updated.trace_id is not None
        assert len(updated.trace_id) > 0

    def test_preserves_an_existing_trace_id(self):
        req = LLMRequest(model="m", prompt="p", trace_id="already-set")
        updated = ensure_trace_id(req)
        assert updated.trace_id == "already-set"

    def test_does_not_mutate_the_original_request(self):
        req = LLMRequest(model="m", prompt="p")
        ensure_trace_id(req)
        assert req.trace_id is None  # pydantic model_copy returns a new object

    def test_generates_distinct_ids_across_calls(self):
        req1 = ensure_trace_id(LLMRequest(model="m", prompt="p"))
        req2 = ensure_trace_id(LLMRequest(model="m", prompt="p"))
        assert req1.trace_id != req2.trace_id


class TestTimer:
    def test_elapsed_ms_is_nonnegative_and_increases(self):
        import time

        timer = Timer()
        first = timer.elapsed_ms
        time.sleep(0.01)
        second = timer.elapsed_ms
        assert first >= 0
        assert second > first


class TestNullMetricsHook:
    def test_on_request_does_nothing_and_does_not_raise(self):
        hook = NullMetricsHook()
        req = LLMRequest(model="m", prompt="p")
        hook.on_request(req, None, None, 1.0)  # must not raise


class TestLoggingMetricsHook:
    def test_logs_success_at_info_level(self, caplog):
        hook = LoggingMetricsHook()
        req = ensure_trace_id(LLMRequest(model="m", prompt="p"))
        resp = LLMResponse(text="hi", model="m", prompt_tokens=3, completion_tokens=2)
        with caplog.at_level(logging.INFO, logger="local_ai_core.runtimes"):
            hook.on_request(req, resp, None, 42.5)
        assert any("llm_request_succeeded" in r.message for r in caplog.records)
        assert any("42.5" in r.message for r in caplog.records)

    def test_logs_failure_at_warning_level(self, caplog):
        hook = LoggingMetricsHook()
        req = ensure_trace_id(LLMRequest(model="m", prompt="p"))
        error = RuntimeUnavailable("connection refused")
        with caplog.at_level(logging.WARNING, logger="local_ai_core.runtimes"):
            hook.on_request(req, None, error, 5.0)
        assert any("llm_request_failed" in r.message for r in caplog.records)
        assert any("RuntimeUnavailable" in r.message for r in caplog.records)

    def test_accepts_a_custom_logger(self):
        custom_logger = logging.getLogger("my.custom.logger")
        hook = LoggingMetricsHook(logger=custom_logger)
        assert hook.logger is custom_logger


class TestWithRetries:
    async def test_returns_result_on_first_success_without_retrying(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            return "ok"

        result = await with_retries(fn, sleep_fn=_no_sleep)
        assert result == "ok"
        assert call_count["n"] == 1

    async def test_retries_on_retryable_error_then_succeeds(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeUnavailable("not up yet")
            return "ok"

        result = await with_retries(fn, max_attempts=5, sleep_fn=_no_sleep)
        assert result == "ok"
        assert call_count["n"] == 3

    async def test_raises_after_exhausting_max_attempts(self):
        async def fn():
            raise RequestTimeout("always times out")

        with pytest.raises(RequestTimeout):
            await with_retries(fn, max_attempts=3, sleep_fn=_no_sleep)

    async def test_does_not_retry_non_retryable_errors(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            raise SchemaValidationError("bad output shape")

        with pytest.raises(SchemaValidationError):
            await with_retries(fn, max_attempts=5, sleep_fn=_no_sleep)
        assert call_count["n"] == 1  # no retry attempted

    async def test_does_not_retry_feature_not_supported(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            raise FeatureNotSupported("grammar not supported by this adapter")

        with pytest.raises(FeatureNotSupported):
            await with_retries(fn, max_attempts=5, sleep_fn=_no_sleep)
        assert call_count["n"] == 1

    async def test_custom_retryable_set_is_honored(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise SchemaValidationError("retryable in this custom config")
            return "ok"

        result = await with_retries(
            fn, max_attempts=3, retryable=(SchemaValidationError,), sleep_fn=_no_sleep
        )
        assert result == "ok"
        assert call_count["n"] == 2

    async def test_rejects_invalid_max_attempts(self):
        async def fn():
            return "ok"

        with pytest.raises(ValueError):
            await with_retries(fn, max_attempts=0, sleep_fn=_no_sleep)

    async def test_uses_exponential_backoff_delays(self):
        delays: list[float] = []

        async def record_sleep(seconds: float) -> None:
            delays.append(seconds)

        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            if call_count["n"] < 4:
                raise RuntimeUnavailable("still down")
            return "ok"

        await with_retries(fn, max_attempts=5, base_delay_seconds=1.0, sleep_fn=record_sleep)
        assert delays == [1.0, 2.0, 4.0]

    async def test_unrelated_exceptions_are_not_caught_at_all(self):
        async def fn():
            raise ValueError("not an LLMError")

        with pytest.raises(ValueError):
            await with_retries(fn, sleep_fn=_no_sleep)


async def _no_sleep(seconds: float) -> None:
    return None
