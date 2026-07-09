"""OllamaRuntime — LLMRuntime adapter for Ollama's native API.

Wraps httpx directly (native /api/generate) rather than an SDK, building on
what Modules 1 and 5 learned exploring Ollama's real behavior, now
normalized to the canonical interface: every public method raises only
LLMError subclasses, response_format is translated or rejected via
FeatureNotSupported, and usage metadata comes from the real runtime
response, never estimated (Module 1 §5's rule).

The payload-building, response-parsing, and error-mapping functions are
pure and unit-tested directly; generate()/stream()/tokenize() are thin
wrappers tested with httpx.MockTransport (see tests/test_ollama_runtime.py
and the theory doc's "Testing strategy" section for what that does and
does not prove).

Enabling this for real (no pip package needed - Ollama is a standalone
server, and this adapter only talks to it over HTTP via `httpx`, already a
project dependency):
    1. `brew install ollama`, then `ollama serve` (or launch the Ollama app)
       to start the local server on `http://localhost:11434` (`DEFAULT_BASE_URL`).
    2. Pull a model, e.g. `ollama pull qwen2.5:1.5b`.
    3. Construct with no fakes - `OllamaRuntime()` (the default `client=None`
       opens a real `httpx.AsyncClient`; only tests pass a
       `client=httpx.AsyncClient(transport=httpx.MockTransport(...))`) - then
       call `await runtime.generate(LLMRequest(model="qwen2.5:1.5b", prompt=...))`.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .base import MetricsHook, NullMetricsHook, Timer, ensure_trace_id
from .errors import (
    FeatureNotSupported,
    InvalidModelResponse,
    LLMError,
    ModelNotLoaded,
    ModelOutOfMemory,
    RequestTimeout,
    RuntimeUnavailable,
)
from .types import LLMRequest, LLMResponse

DEFAULT_BASE_URL = "http://localhost:11434"


def build_generate_payload(request: LLMRequest) -> dict[str, Any]:
    """Pure translation: LLMRequest -> Ollama /api/generate JSON body.

    Raises FeatureNotSupported for response_format types Ollama's native API
    cannot honor - confirmed by Module 5's feature_matrix.py: Ollama has no
    user-facing grammar support.
    """
    if request.response_format.type == "grammar":
        raise FeatureNotSupported(
            "Ollama does not support grammar-constrained decoding "
            "(see scripts/module_05/feature_matrix.py)"
        )

    payload: dict[str, Any] = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
        "options": {"temperature": request.temperature, "num_predict": request.max_tokens},
    }
    if request.system:
        payload["system"] = request.system
    if request.stop:
        payload["options"]["stop"] = request.stop
    if request.response_format.type == "json_schema":
        payload["format"] = request.response_format.schema_ or "json"
    return payload


def parse_generate_response(request: LLMRequest, data: dict[str, Any], latency_ms: float) -> LLMResponse:
    """Pure translation: Ollama /api/generate JSON body -> LLMResponse."""
    if "response" not in data:
        raise InvalidModelResponse(f"Ollama response missing 'response' field: {data!r}")
    return LLMResponse(
        text=data["response"],
        model=request.model,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        latency_ms=latency_ms,
        stop_reason="stop" if data.get("done") else None,
        raw=data,
    )


def map_httpx_error(exc: httpx.HTTPError, *, model: str) -> LLMError:
    """Pure translation: httpx exception -> the LLMError taxonomy.

    This is what resolves the failure_rate/timeout_rate gap flagged in
    Modules 4 and 5's reports: httpx's own exception hierarchy already
    distinguishes "couldn't even connect" from "connected but timed out
    waiting," so this maps that distinction onto our taxonomy instead of
    collapsing everything into one generic error, the way Module 1's
    lab-local ollama_probe.py did.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return RuntimeUnavailable(f"Could not connect to Ollama: {exc}", cause=exc)
    if isinstance(exc, httpx.TimeoutException):
        return RequestTimeout(f"Request to Ollama timed out: {exc}", cause=exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 404:
            return ModelNotLoaded(f"Model '{model}' not found on Ollama: {exc}", cause=exc)
        if status == 500:
            body = exc.response.text.lower()
            if "out of memory" in body or "oom" in body:
                return ModelOutOfMemory(f"Ollama reported an out-of-memory condition: {exc}", cause=exc)
            return InvalidModelResponse(f"Ollama returned a server error: {exc}", cause=exc)
        return InvalidModelResponse(f"Ollama returned an unexpected status {status}: {exc}", cause=exc)
    return RuntimeUnavailable(f"Could not reach Ollama: {exc}", cause=exc)


class OllamaRuntime:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self.metrics_hook: MetricsHook = metrics_hook or NullMetricsHook()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        request = ensure_trace_id(request)
        payload = build_generate_payload(request)  # may raise FeatureNotSupported, no network yet
        timer = Timer()
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            error = map_httpx_error(exc, model=request.model)
            self.metrics_hook.on_request(request, None, error, timer.elapsed_ms)
            raise error from exc

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            error = InvalidModelResponse(f"Ollama returned non-JSON response: {exc}", cause=exc)
            self.metrics_hook.on_request(request, None, error, timer.elapsed_ms)
            raise error from exc

        response = parse_generate_response(request, data, timer.elapsed_ms)
        self.metrics_hook.on_request(request, response, None, timer.elapsed_ms)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        request = ensure_trace_id(request)
        payload = build_generate_payload(request)
        payload["stream"] = True
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise InvalidModelResponse(
                            f"Ollama stream returned non-JSON line: {exc}", cause=exc
                        ) from exc
                    text = data.get("response", "")
                    if text:
                        yield text
        except httpx.HTTPError as exc:
            raise map_httpx_error(exc, model=request.model) from exc

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        # Ollama has no stable, documented pre-flight tokenize endpoint
        # (scripts/module_05/feature_matrix.py marks this "partial") - raise
        # rather than approximate. Callers needing pre-flight counts should
        # use scripts/module_01/token_estimate.HFTokenizerCounter instead,
        # or read prompt_eval_count from a generate() response post-hoc.
        raise FeatureNotSupported(
            "Ollama has no stable pre-flight tokenize endpoint; use "
            "token_estimate.HFTokenizerCounter (Module 1) for pre-flight counting, "
            "or read LLMResponse.prompt_tokens from generate() for a post-hoc count."
        )
