"""The canonical LLMRuntime Protocol, plus the shared retry/metrics/trace-id
infrastructure every adapter uses (curriculum.md §16).

This is THE definition of the runtime abstraction for the whole course.
Earlier modules smoke-tested runtimes with lab-local code (Modules 1-5);
later modules extend serving behavior; nothing may redefine this Protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator, Awaitable, Callable, Protocol, TypeVar

from .errors import LLMError, RequestTimeout, RuntimeUnavailable
from .types import LLMRequest, LLMResponse


class LLMRuntime(Protocol):
    """Structural interface every runtime adapter implements.

    Protocol (not ABC): any object with matching async methods satisfies
    this without explicit inheritance - useful for FakeRuntime and any
    future test doubles that don't want a real base-class dependency.
    """

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]: ...

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]: ...


# --- Trace IDs (Lab 7) -------------------------------------------------


def ensure_trace_id(request: LLMRequest) -> LLMRequest:
    """Return a copy of ``request`` with a trace_id filled in if it's missing.

    Every adapter should call this before processing a request so every
    request is traceable end to end, even when the caller didn't set one.
    """
    if request.trace_id:
        return request
    return request.model_copy(update={"trace_id": str(uuid.uuid4())})


# --- Metrics hooks (Labs 3, 9) ------------------------------------------


class MetricsHook(Protocol):
    """Invoked once per request, success or failure."""

    def on_request(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        error: LLMError | None,
        latency_ms: float,
    ) -> None: ...


class NullMetricsHook:
    """No-op hook - the default for adapters that don't need metrics."""

    def on_request(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        error: LLMError | None,
        latency_ms: float,
    ) -> None:
        return None


class LoggingMetricsHook:
    """Structured logging via the stdlib ``logging`` module.

    Satisfies both Lab 3 (structured logging) and Lab 9 (metrics hooks) with
    one small piece of infrastructure - a metrics hook that logs *is*
    structured logging.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("local_ai_core.runtimes")

    def on_request(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        error: LLMError | None,
        latency_ms: float,
    ) -> None:
        fields = {
            "trace_id": request.trace_id,
            "model": request.model,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": response.prompt_tokens if response else None,
            "completion_tokens": response.completion_tokens if response else None,
            "error_type": type(error).__name__ if error else None,
        }
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        if error is not None:
            self.logger.warning("llm_request_failed %s", rendered)
        else:
            self.logger.info("llm_request_succeeded %s", rendered)


class Timer:
    """Small context-manager-free timer helper so adapters measure latency
    consistently: ``timer = Timer(); ...; timer.elapsed_ms``.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


# --- Retries (Labs 4, 5) --------------------------------------------------

T = TypeVar("T")

DEFAULT_RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (RuntimeUnavailable, RequestTimeout)


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    retryable: tuple[type[LLMError], ...] = DEFAULT_RETRYABLE_ERRORS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Retry ``fn`` with exponential backoff, but ONLY for exception types in
    ``retryable``. Everything else (a validation failure, an unsupported
    feature request) propagates on the first attempt - retrying a
    deterministic failure just wastes time and obscures the real error
    (theory doc Lab 5: "no-retry for deterministic validation failures").

    ``sleep_fn`` is injectable so tests can run this with zero real delay.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: LLMError | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
            await sleep_fn(base_delay_seconds * (2**attempt))
    # Unreachable in practice (the loop always returns or raises), but keeps
    # type checkers happy and fails loudly instead of returning None if it
    # somehow is reached.
    assert last_error is not None
    raise last_error
