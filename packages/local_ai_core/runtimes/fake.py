"""FakeRuntime — a fully in-memory LLMRuntime for deterministic unit tests
(Lab 2). Every other module's tests that need "a model" should use this
instead of mocking a real runtime ad hoc.

Configurable to fail in specific, controlled ways so retry logic (Lab 4/5)
can be tested without timing games against a real flaky server.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import MetricsHook, NullMetricsHook, Timer
from .errors import FeatureNotSupported, LLMError, RuntimeUnavailable
from .types import LLMRequest, LLMResponse


class FakeRuntime:
    """Implements the LLMRuntime protocol structurally (no inheritance needed).

    - ``responses``: per-model canned response text; falls back to
      ``default_response`` for any model not listed.
    - ``fail_with``: if set, every ``generate()`` call raises this error.
    - ``fail_first_n_calls`` + ``transient_error_for_first_n``: raise a
      transient error for exactly the first N calls, then succeed - built
      specifically to test ``with_retries`` deterministically.
    - ``requests_received``/``call_count``: inspectable call history for
      assertions like "the retry wrapper called generate exactly 3 times."
    """

    def __init__(
        self,
        *,
        responses: dict[str, str] | None = None,
        default_response: str = "This is a fake response.",
        fail_with: LLMError | None = None,
        fail_first_n_calls: int = 0,
        transient_error_for_first_n: LLMError | None = None,
        simulated_latency_ms: float = 0.0,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default_response = default_response
        self.fail_with = fail_with
        self.fail_first_n_calls = fail_first_n_calls
        self.transient_error_for_first_n = transient_error_for_first_n or RuntimeUnavailable(
            "FakeRuntime: simulated transient failure"
        )
        self.simulated_latency_ms = simulated_latency_ms
        self.metrics_hook: MetricsHook = metrics_hook or NullMetricsHook()
        self.call_count = 0
        self.requests_received: list[LLMRequest] = []

    def _next_error(self, request: LLMRequest) -> LLMError | None:
        if self.fail_first_n_calls and self.call_count <= self.fail_first_n_calls:
            return self.transient_error_for_first_n
        if self.fail_with is not None:
            return self.fail_with
        if request.response_format.type == "grammar":
            return FeatureNotSupported("FakeRuntime does not support grammar-constrained decoding")
        return None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        self.requests_received.append(request)
        timer = Timer()
        if self.simulated_latency_ms:
            await asyncio.sleep(self.simulated_latency_ms / 1000)

        error = self._next_error(request)
        if error is not None:
            self.metrics_hook.on_request(request, None, error, timer.elapsed_ms)
            raise error

        text = self.responses.get(request.model, self.default_response)
        response = LLMResponse(
            text=text,
            model=request.model,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=timer.elapsed_ms,
            stop_reason="stop",
        )
        self.metrics_hook.on_request(request, response, None, timer.elapsed_ms)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        error = self._next_error(request)
        if error is not None:
            self.call_count += 1
            self.requests_received.append(request)
            raise error

        self.call_count += 1
        self.requests_received.append(request)
        text = self.responses.get(request.model, self.default_response)
        for word in text.split():
            await asyncio.sleep(0)  # keep this a real async generator, not a sync loop
            yield word + " "

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        # A fake, deterministic "tokenizer" - one fake id per whitespace-split
        # word. Never used to make real token-budget decisions (Module 1 §5);
        # only for exercising code paths that call tokenize().
        return [abs(hash(word)) % 50_000 for word in rendered_prompt.split()]
