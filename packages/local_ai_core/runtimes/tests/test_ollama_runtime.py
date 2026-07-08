import json

import httpx
import pytest

from local_ai_core.runtimes.errors import (
    FeatureNotSupported,
    InvalidModelResponse,
    ModelNotLoaded,
    ModelOutOfMemory,
    RequestTimeout,
    RuntimeUnavailable,
)
from local_ai_core.runtimes.ollama import (
    OllamaRuntime,
    build_generate_payload,
    map_httpx_error,
    parse_generate_response,
)
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat

# --- Pure translation function tests (no network at all) -------------------


class TestBuildGeneratePayload:
    def test_minimal_request_builds_expected_payload(self):
        req = LLMRequest(model="qwen2.5:1.5b", prompt="hi")
        payload = build_generate_payload(req)
        assert payload["model"] == "qwen2.5:1.5b"
        assert payload["prompt"] == "hi"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.0

    def test_includes_system_prompt_when_present(self):
        req = LLMRequest(model="m", prompt="p", system="Be terse.")
        payload = build_generate_payload(req)
        assert payload["system"] == "Be terse."

    def test_omits_system_when_absent(self):
        req = LLMRequest(model="m", prompt="p")
        payload = build_generate_payload(req)
        assert "system" not in payload

    def test_includes_stop_sequences_in_options(self):
        req = LLMRequest(model="m", prompt="p", stop=["\n\n", "END"])
        payload = build_generate_payload(req)
        assert payload["options"]["stop"] == ["\n\n", "END"]

    def test_json_schema_response_format_sets_format_field(self):
        req = LLMRequest(
            model="m", prompt="p",
            response_format=ResponseFormat(type="json_schema", schema={"type": "object"}),
        )
        payload = build_generate_payload(req)
        assert payload["format"] == {"type": "object"}

    def test_json_schema_without_explicit_schema_defaults_to_json_string(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="json_schema"))
        payload = build_generate_payload(req)
        assert payload["format"] == "json"

    def test_grammar_response_format_raises_feature_not_supported(self):
        req = LLMRequest(
            model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="root ::= 'x'")
        )
        with pytest.raises(FeatureNotSupported):
            build_generate_payload(req)


class TestParseGenerateResponse:
    def test_parses_a_normal_response(self):
        req = LLMRequest(model="qwen2.5:1.5b", prompt="p")
        data = {"response": "hello", "done": True, "prompt_eval_count": 5, "eval_count": 3}
        response = parse_generate_response(req, data, latency_ms=42.0)
        assert response.text == "hello"
        assert response.model == "qwen2.5:1.5b"
        assert response.prompt_tokens == 5
        assert response.completion_tokens == 3
        assert response.latency_ms == pytest.approx(42.0)
        assert response.stop_reason == "stop"

    def test_raw_field_preserves_full_response(self):
        req = LLMRequest(model="m", prompt="p")
        data = {"response": "hi", "done": True, "extra_field": "xyz"}
        response = parse_generate_response(req, data, latency_ms=1.0)
        assert response.raw == data

    def test_missing_response_field_raises_invalid_model_response(self):
        req = LLMRequest(model="m", prompt="p")
        with pytest.raises(InvalidModelResponse):
            parse_generate_response(req, {"done": True}, latency_ms=1.0)

    def test_not_done_has_no_stop_reason(self):
        req = LLMRequest(model="m", prompt="p")
        response = parse_generate_response(req, {"response": "partial", "done": False}, latency_ms=1.0)
        assert response.stop_reason is None


class TestMapHttpxError:
    def test_connect_error_maps_to_runtime_unavailable(self):
        exc = httpx.ConnectError("connection refused")
        assert isinstance(map_httpx_error(exc, model="m"), RuntimeUnavailable)

    def test_connect_timeout_maps_to_runtime_unavailable(self):
        # ConnectTimeout is technically a TimeoutException subclass, but
        # semantically "couldn't even establish a connection" - mapped to
        # RuntimeUnavailable, not RequestTimeout (theory doc §6 error taxonomy).
        exc = httpx.ConnectTimeout("timed out connecting")
        assert isinstance(map_httpx_error(exc, model="m"), RuntimeUnavailable)

    def test_read_timeout_maps_to_request_timeout(self):
        exc = httpx.ReadTimeout("timed out reading response")
        assert isinstance(map_httpx_error(exc, model="m"), RequestTimeout)

    def test_pool_timeout_maps_to_request_timeout(self):
        exc = httpx.PoolTimeout("timed out waiting for a connection")
        assert isinstance(map_httpx_error(exc, model="m"), RequestTimeout)

    def test_404_status_maps_to_model_not_loaded(self):
        request = httpx.Request("POST", "http://localhost:11434/api/generate")
        response = httpx.Response(404, request=request, text="model not found")
        exc = httpx.HTTPStatusError("404", request=request, response=response)
        assert isinstance(map_httpx_error(exc, model="missing-model"), ModelNotLoaded)

    def test_500_with_oom_body_maps_to_model_out_of_memory(self):
        request = httpx.Request("POST", "http://localhost:11434/api/generate")
        response = httpx.Response(500, request=request, text="CUDA error: out of memory")
        exc = httpx.HTTPStatusError("500", request=request, response=response)
        assert isinstance(map_httpx_error(exc, model="m"), ModelOutOfMemory)

    def test_500_without_oom_body_maps_to_invalid_model_response(self):
        request = httpx.Request("POST", "http://localhost:11434/api/generate")
        response = httpx.Response(500, request=request, text="internal server error")
        exc = httpx.HTTPStatusError("500", request=request, response=response)
        assert isinstance(map_httpx_error(exc, model="m"), InvalidModelResponse)

    def test_error_preserves_original_exception_as_cause(self):
        exc = httpx.ConnectError("refused")
        mapped = map_httpx_error(exc, model="m")
        assert mapped.cause is exc


# --- MockTransport-backed adapter tests (see theory doc "Testing strategy") -


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOllamaRuntimeGenerate:
    async def test_successful_generate_returns_llm_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/generate"
            body = json.loads(request.content)
            assert body["model"] == "qwen2.5:1.5b"
            return httpx.Response(
                200,
                json={"response": "Hello!", "done": True, "prompt_eval_count": 4, "eval_count": 2},
            )

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        response = await runtime.generate(LLMRequest(model="qwen2.5:1.5b", prompt="Say hi"))
        assert response.text == "Hello!"
        assert response.prompt_tokens == 4
        assert response.completion_tokens == 2

    async def test_generate_fills_in_a_trace_id(self):
        seen_trace_ids = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "ok", "done": True})

        class RecordingHook:
            def on_request(self, request, response, error, latency_ms):
                seen_trace_ids.append(request.trace_id)

        runtime = OllamaRuntime(client=_client_with_handler(handler), metrics_hook=RecordingHook())
        await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert seen_trace_ids[0] is not None

    async def test_connection_refused_raises_runtime_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_read_timeout_raises_request_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        with pytest.raises(RequestTimeout):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_model_not_found_raises_model_not_loaded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="model not found")

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        with pytest.raises(ModelNotLoaded):
            await runtime.generate(LLMRequest(model="does-not-exist", prompt="p"))

    async def test_grammar_request_never_hits_the_network(self):
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("grammar requests must be rejected before any network call")

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="x"))
        with pytest.raises(FeatureNotSupported):
            await runtime.generate(request)

    async def test_metrics_hook_is_called_on_success(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "ok", "done": True})

        class RecordingHook:
            def on_request(self, request, response, error, latency_ms):
                calls.append((response, error))

        runtime = OllamaRuntime(client=_client_with_handler(handler), metrics_hook=RecordingHook())
        await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert len(calls) == 1
        assert calls[0][0] is not None
        assert calls[0][1] is None

    async def test_metrics_hook_is_called_on_failure(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        class RecordingHook:
            def on_request(self, request, response, error, latency_ms):
                calls.append((response, error))

        runtime = OllamaRuntime(client=_client_with_handler(handler), metrics_hook=RecordingHook())
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert len(calls) == 1
        assert calls[0][0] is None
        assert isinstance(calls[0][1], RuntimeUnavailable)


class TestOllamaRuntimeStream:
    async def test_stream_yields_text_fragments(self):
        ndjson = "\n".join(
            [
                json.dumps({"response": "Hello", "done": False}),
                json.dumps({"response": ", world", "done": False}),
                json.dumps({"response": "", "done": True, "eval_count": 2}),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=ndjson.encode())

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        chunks = [c async for c in runtime.stream(LLMRequest(model="m", prompt="p"))]
        assert "".join(chunks) == "Hello, world"

    async def test_stream_connection_error_raises_runtime_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        runtime = OllamaRuntime(client=_client_with_handler(handler))
        with pytest.raises(RuntimeUnavailable):
            async for _ in runtime.stream(LLMRequest(model="m", prompt="p")):
                pass


class TestOllamaRuntimeTokenize:
    async def test_tokenize_always_raises_feature_not_supported(self):
        runtime = OllamaRuntime()
        with pytest.raises(FeatureNotSupported):
            await runtime.tokenize("m", "some rendered prompt")
