"""OpenAICompatibleRuntime — LLMRuntime adapter for llama.cpp-family
OpenAI-compatible servers (llama-server, llama-cpp-python[server]), using
the standard `openai` Python client (Module 5 theory doc §4: the same
client code path an application would use against a hosted API, pointed at
a local server instead).

`generate`/`stream` go through the OpenAI client (mockable via its
`http_client` constructor argument); `tokenize` calls llama.cpp's native
`/tokenize` endpoint directly with httpx, since that endpoint is a
llama.cpp-family extension, not part of the OpenAI API surface (confirmed
"yes" in scripts/module_05/feature_matrix.py for llama.cpp/llama-cpp-python,
unlike Ollama).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx
import openai
from openai import AsyncOpenAI

from .base import MetricsHook, NullMetricsHook, Timer, ensure_trace_id
from .errors import (
    ContextTooLarge,
    FeatureNotSupported,
    InvalidModelResponse,
    LLMError,
    ModelNotLoaded,
    RequestTimeout,
    RuntimeUnavailable,
)
from .types import LLMRequest, LLMResponse

DEFAULT_BASE_URL = "http://localhost:8080/v1"


def build_chat_messages(request: LLMRequest) -> list[dict[str, str]]:
    """Pure translation: LLMRequest -> OpenAI chat messages list."""
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    return messages


def build_extra_body(request: LLMRequest) -> dict[str, Any]:
    """Pure translation: response_format -> llama.cpp-server-specific extra
    request body fields not part of the standard OpenAI API (the `grammar`
    field is a llama.cpp/llama-cpp-python extension).
    """
    if request.response_format.type != "grammar":
        return {}
    if not request.response_format.grammar:
        raise FeatureNotSupported("grammar response_format requested but no grammar string was provided")
    return {"grammar": request.response_format.grammar}


def build_openai_response_format(request: LLMRequest) -> dict[str, Any] | None:
    """Pure translation: response_format -> the standard OpenAI response_format param."""
    if request.response_format.type != "json_schema":
        return None
    schema = request.response_format.schema_
    if schema:
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": True}}
    return {"type": "json_object"}


def map_openai_error(exc: Exception, *, model: str) -> LLMError:
    """Pure translation: an openai-SDK exception -> the LLMError taxonomy."""
    if isinstance(exc, openai.APITimeoutError):
        return RequestTimeout(f"Request to the OpenAI-compatible server timed out: {exc}", cause=exc)
    if isinstance(exc, openai.APIConnectionError):
        return RuntimeUnavailable(f"Could not connect to the OpenAI-compatible server: {exc}", cause=exc)
    if isinstance(exc, openai.NotFoundError):
        return ModelNotLoaded(f"Model '{model}' not found: {exc}", cause=exc)
    if isinstance(exc, openai.BadRequestError):
        message = str(exc).lower()
        if "context" in message and ("exceed" in message or "too long" in message or "too many tokens" in message):
            return ContextTooLarge(f"Prompt exceeds the server's context window: {exc}", cause=exc)
        return InvalidModelResponse(f"Server rejected the request: {exc}", cause=exc)
    if isinstance(exc, openai.APIStatusError):
        return InvalidModelResponse(f"Server returned an error: {exc}", cause=exc)
    return RuntimeUnavailable(f"Unexpected error calling the OpenAI-compatible server: {exc}", cause=exc)


class OpenAICompatibleRuntime:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        # max_retries=0: retry policy lives in exactly one place, base.py's
        # with_retries(), applied by callers around generate()/stream(). If
        # the SDK also retried internally, a caller retrying on
        # RuntimeUnavailable would compose with the SDK's own retries,
        # multiplying attempts and hiding real latency.
        self._openai_client = openai_client or AsyncOpenAI(
            base_url=base_url, api_key="not-needed-for-local-server", timeout=timeout, max_retries=0
        )
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_http_client = http_client is None
        self.metrics_hook: MetricsHook = metrics_hook or NullMetricsHook()

    async def aclose(self) -> None:
        await self._openai_client.close()
        if self._owns_http_client:
            await self._http_client.aclose()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        request = ensure_trace_id(request)
        extra_body = build_extra_body(request)  # may raise FeatureNotSupported, no network yet
        timer = Timer()
        try:
            completion = await self._openai_client.chat.completions.create(
                model=request.model,
                messages=build_chat_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or None,
                response_format=build_openai_response_format(request),
                extra_body=extra_body or None,
            )
        except openai.OpenAIError as exc:
            error = map_openai_error(exc, model=request.model)
            self.metrics_hook.on_request(request, None, error, timer.elapsed_ms)
            raise error from exc

        choice = completion.choices[0]
        usage = completion.usage
        response = LLMResponse(
            text=choice.message.content or "",
            model=request.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=timer.elapsed_ms,
            stop_reason=choice.finish_reason,
            raw=completion.model_dump(),
        )
        self.metrics_hook.on_request(request, response, None, timer.elapsed_ms)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        request = ensure_trace_id(request)
        extra_body = build_extra_body(request)
        try:
            stream = await self._openai_client.chat.completions.create(
                model=request.model,
                messages=build_chat_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or None,
                extra_body=extra_body or None,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta_text = chunk.choices[0].delta.content
                if delta_text:
                    yield delta_text
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, model=request.model) from exc

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        native_base = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            resp = await self._http_client.post(
                f"{native_base}/tokenize", json={"content": rendered_prompt}, timeout=self.timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                raise RuntimeUnavailable(f"Could not connect to the server: {exc}", cause=exc) from exc
            if isinstance(exc, httpx.TimeoutException):
                raise RequestTimeout(f"Tokenize request timed out: {exc}", cause=exc) from exc
            raise InvalidModelResponse(f"Tokenize request failed: {exc}", cause=exc) from exc

        data = resp.json()
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise InvalidModelResponse(f"Unexpected /tokenize response shape: {data!r}")
        return tokens
