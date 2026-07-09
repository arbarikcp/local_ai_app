"""FallbackRuntime — an ordered chain of LLMRuntime adapters (theory doc
§11). `generate()` tries each runtime in order, moving to the next only on
the same retryable error types Module 6's `with_retries()` already
recognizes (`RuntimeUnavailable`, `RequestTimeout`) - a non-retryable error
(a validation failure, an unsupported feature) propagates immediately
without wasting a fallback attempt on a deterministic failure that every
runtime in the chain would also raise.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.errors import LLMError, RequestTimeout, RuntimeUnavailable
from local_ai_core.runtimes.types import LLMRequest, LLMResponse

DEFAULT_FALLBACK_ERRORS: tuple[type[LLMError], ...] = (RuntimeUnavailable, RequestTimeout)


@dataclass(frozen=True)
class FallbackResult:
    response: LLMResponse
    runtime_index: int
    attempts: int


class NoRuntimesAvailable(LLMError):
    """Every runtime in the fallback chain failed with a retryable error."""


class FallbackRuntime:
    def __init__(
        self,
        runtimes: list[LLMRuntime],
        *,
        fallback_errors: tuple[type[LLMError], ...] = DEFAULT_FALLBACK_ERRORS,
    ) -> None:
        if not runtimes:
            raise ValueError("runtimes must not be empty")
        self._runtimes = runtimes
        self._fallback_errors = fallback_errors

    async def generate(self, request: LLMRequest) -> FallbackResult:
        last_error: LLMError | None = None
        for index, runtime in enumerate(self._runtimes):
            try:
                response = await runtime.generate(request)
                return FallbackResult(response=response, runtime_index=index, attempts=index + 1)
            except self._fallback_errors as exc:
                last_error = exc
                continue

        raise NoRuntimesAvailable(
            f"All {len(self._runtimes)} runtimes in the fallback chain failed", cause=last_error
        )
