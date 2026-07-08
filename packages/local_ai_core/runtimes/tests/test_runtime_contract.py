"""Shared contract test suite, parametrized across every LLMRuntime adapter.

This is the curriculum's explicit ask (curriculum.md §16 deliverable list):
one set of assertions proving the abstraction actually abstracts - that
FakeRuntime, OllamaRuntime, OpenAICompatibleRuntime, and MLXRuntime are
interchangeable from a caller's point of view - rather than each adapter
only being tested in isolation against its own expectations.

Ollama and OpenAICompatibleRuntime are backed by httpx.MockTransport;
MLXRuntime is backed by injected fakes. See the theory doc's "Testing
strategy" section for exactly what that does and does not prove.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from local_ai_core.runtimes.errors import LLMError
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.mlx import MLXRuntime
from local_ai_core.runtimes.ollama import OllamaRuntime
from local_ai_core.runtimes.openai_compatible import OpenAICompatibleRuntime
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat


def _make_fake_runtime() -> FakeRuntime:
    return FakeRuntime(default_response="Hello from the fake runtime")


def _make_ollama_runtime() -> OllamaRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            ndjson = "\n".join(
                [
                    json.dumps({"response": "Hello ", "done": False}),
                    json.dumps({"response": "from ollama", "done": False}),
                    json.dumps({"response": "", "done": True, "eval_count": 3}),
                ]
            )
            return httpx.Response(200, content=ndjson.encode())
        return httpx.Response(
            200, json={"response": "Hello from ollama", "done": True, "prompt_eval_count": 3, "eval_count": 3}
        )

    return OllamaRuntime(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _make_openai_compatible_runtime() -> OpenAICompatibleRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            sse = (
                'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
                '"choices":[{"index":0,"delta":{"content":"Hello from openai-compatible"},'
                '"finish_reason":null}]}\n\n'
                'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from openai-compatible"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            },
        )

    openai_client = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAICompatibleRuntime(openai_client=openai_client)


class _FakeMlxTokenizer:
    def apply_chat_template(self, messages: list[dict], tokenize: bool, add_generation_prompt: bool) -> str:
        return "rendered prompt"

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]


@dataclass(frozen=True)
class _FakeMlxStreamItem:
    text: str


def _make_mlx_runtime() -> MLXRuntime:
    def load_fn(model_id: str) -> tuple[Any, Any]:
        return ("fake-mlx-model", _FakeMlxTokenizer())

    def generate_fn(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> str:
        return "Hello from mlx"

    def stream_generate_fn(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int):
        for word in ["Hello", " from", " mlx"]:
            yield _FakeMlxStreamItem(text=word)

    return MLXRuntime(load_fn=load_fn, generate_fn=generate_fn, stream_generate_fn=stream_generate_fn)


RUNTIME_FACTORIES = {
    "fake": _make_fake_runtime,
    "ollama": _make_ollama_runtime,
    "openai_compatible": _make_openai_compatible_runtime,
    "mlx": _make_mlx_runtime,
}


# Per-adapter "a response_format this adapter genuinely does not support."
# Deliberately NOT uniform across adapters: OpenAICompatibleRuntime (real
# llama.cpp/llama-cpp-python servers) actually supports both json_schema
# AND grammar (confirmed in scripts/module_05/feature_matrix.py), so there
# is no ResponseFormat it should reject - that capability difference is
# Module 5's entire point, and an earlier version of this test asserted
# "every adapter rejects grammar" uniformly, which this test suite itself
# caught as false the first time it ran against OpenAICompatibleRuntime.
UNSUPPORTED_FORMAT_BY_ADAPTER: dict[str, ResponseFormat] = {
    "fake": ResponseFormat(type="grammar", grammar="root ::= 'x'"),
    "ollama": ResponseFormat(type="grammar", grammar="root ::= 'x'"),
    "mlx": ResponseFormat(type="json_schema"),
}


@pytest.fixture(params=list(RUNTIME_FACTORIES.keys()))
def runtime_case(request):
    return request.param, RUNTIME_FACTORIES[request.param]()


class TestRuntimeContract:
    async def test_generate_returns_llm_response_with_nonempty_text_and_matching_model(self, runtime_case):
        _name, runtime = runtime_case
        request = LLMRequest(model="contract-test-model", prompt="hi")
        response = await runtime.generate(request)
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        assert response.model == "contract-test-model"

    async def test_generate_latency_is_recorded_and_nonnegative(self, runtime_case):
        _name, runtime = runtime_case
        response = await runtime.generate(LLMRequest(model="m", prompt="hi"))
        assert response.latency_ms is not None
        assert response.latency_ms >= 0

    async def test_stream_yields_only_nonempty_strings_in_order(self, runtime_case):
        _name, runtime = runtime_case
        chunks = [c async for c in runtime.stream(LLMRequest(model="m", prompt="hi"))]
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    async def test_tokenize_returns_list_of_ints_or_a_documented_feature_not_supported(self, runtime_case):
        _name, runtime = runtime_case
        # OllamaRuntime intentionally raises FeatureNotSupported here (no
        # stable pre-flight tokenize endpoint - see ollama.py's docstring).
        # That is still contract-compliant: an LLMError, not a crash or a
        # silently wrong answer. Every other adapter returns real token ids.
        try:
            tokens = await runtime.tokenize("m", "hello world")
        except LLMError:
            return
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    async def test_unsupported_response_format_raises_llm_error_not_a_raw_runtime_exception(self, runtime_case):
        name, runtime = runtime_case
        if name not in UNSUPPORTED_FORMAT_BY_ADAPTER:
            pytest.skip(f"{name} genuinely supports every ResponseFormatType - nothing to reject")
        request = LLMRequest(model="m", prompt="hi", response_format=UNSUPPORTED_FORMAT_BY_ADAPTER[name])
        with pytest.raises(LLMError):
            await runtime.generate(request)

    async def test_response_format_rejection_happens_before_stream_yields_anything(self, runtime_case):
        name, runtime = runtime_case
        if name not in UNSUPPORTED_FORMAT_BY_ADAPTER:
            pytest.skip(f"{name} genuinely supports every ResponseFormatType - nothing to reject")
        request = LLMRequest(model="m", prompt="hi", response_format=UNSUPPORTED_FORMAT_BY_ADAPTER[name])
        with pytest.raises(LLMError):
            async for _ in runtime.stream(request):
                pytest.fail("must not yield any chunk for an unsupported response_format")
