import httpx
import openai
import pytest
from openai import AsyncOpenAI

from local_ai_core.runtimes.errors import (
    ContextTooLarge,
    FeatureNotSupported,
    InvalidModelResponse,
    ModelNotLoaded,
    RequestTimeout,
    RuntimeUnavailable,
)
from local_ai_core.runtimes.openai_compatible import (
    OpenAICompatibleRuntime,
    build_chat_messages,
    build_extra_body,
    build_openai_response_format,
    map_openai_error,
)
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat

# --- Pure translation function tests (no network at all) -------------------


class TestBuildChatMessages:
    def test_user_only_when_no_system_prompt(self):
        messages = build_chat_messages(LLMRequest(model="m", prompt="hi"))
        assert messages == [{"role": "user", "content": "hi"}]

    def test_includes_system_message_when_present(self):
        messages = build_chat_messages(LLMRequest(model="m", prompt="hi", system="Be terse."))
        assert messages[0] == {"role": "system", "content": "Be terse."}
        assert messages[1] == {"role": "user", "content": "hi"}


class TestBuildExtraBody:
    def test_empty_for_text_format(self):
        assert build_extra_body(LLMRequest(model="m", prompt="p")) == {}

    def test_empty_for_json_schema_format(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="json_schema"))
        assert build_extra_body(req) == {}

    def test_includes_grammar_string_for_grammar_format(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="root ::= 'x'"))
        assert build_extra_body(req) == {"grammar": "root ::= 'x'"}

    def test_grammar_format_without_grammar_string_raises(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar"))
        with pytest.raises(FeatureNotSupported):
            build_extra_body(req)


class TestBuildOpenaiResponseFormat:
    def test_none_for_text(self):
        assert build_openai_response_format(LLMRequest(model="m", prompt="p")) is None

    def test_none_for_grammar(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="x"))
        assert build_openai_response_format(req) is None

    def test_json_object_when_no_schema_given(self):
        req = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="json_schema"))
        assert build_openai_response_format(req) == {"type": "json_object"}

    def test_json_schema_wraps_the_provided_schema(self):
        req = LLMRequest(
            model="m", prompt="p", response_format=ResponseFormat(type="json_schema", schema={"type": "object"})
        )
        result = build_openai_response_format(req)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["schema"] == {"type": "object"}


class TestMapOpenaiError:
    def test_api_connection_error_maps_to_runtime_unavailable(self):
        request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        exc = openai.APIConnectionError(request=request)
        assert isinstance(map_openai_error(exc, model="m"), RuntimeUnavailable)

    def test_api_timeout_error_maps_to_request_timeout(self):
        request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        exc = openai.APITimeoutError(request=request)
        assert isinstance(map_openai_error(exc, model="m"), RequestTimeout)

    def test_not_found_error_maps_to_model_not_loaded(self):
        request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        response = httpx.Response(404, request=request, json={"error": {"message": "not found"}})
        exc = openai.NotFoundError("not found", response=response, body=None)
        assert isinstance(map_openai_error(exc, model="m"), ModelNotLoaded)

    def test_bad_request_with_context_message_maps_to_context_too_large(self):
        request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        response = httpx.Response(400, request=request, json={"error": {"message": "context length exceeded"}})
        exc = openai.BadRequestError("context length exceeded", response=response, body=None)
        assert isinstance(map_openai_error(exc, model="m"), ContextTooLarge)

    def test_bad_request_without_context_message_maps_to_invalid_model_response(self):
        request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        response = httpx.Response(400, request=request, json={"error": {"message": "malformed request"}})
        exc = openai.BadRequestError("malformed request", response=response, body=None)
        assert isinstance(map_openai_error(exc, model="m"), InvalidModelResponse)


# --- MockTransport-backed adapter tests -------------------------------------


def _completion_json(text: str = "Hello!", finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "local-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _runtime_with_handler(handler, *, http_handler=None) -> OpenAICompatibleRuntime:
    openai_client = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(http_handler or handler))
    return OpenAICompatibleRuntime(openai_client=openai_client, http_client=http_client)


class TestOpenAICompatibleRuntimeGenerate:
    async def test_successful_generate_returns_llm_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json=_completion_json("Hello there!"))

        runtime = _runtime_with_handler(handler)
        response = await runtime.generate(LLMRequest(model="local-model", prompt="hi"))
        assert response.text == "Hello there!"
        assert response.prompt_tokens == 5
        assert response.completion_tokens == 2
        assert response.stop_reason == "stop"

    async def test_connection_error_raises_runtime_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        runtime = _runtime_with_handler(handler)
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_model_not_found_raises_model_not_loaded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"message": "no such model"}})

        runtime = _runtime_with_handler(handler)
        with pytest.raises(ModelNotLoaded):
            await runtime.generate(LLMRequest(model="missing", prompt="p"))

    async def test_grammar_without_string_never_hits_the_network(self):
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("must be rejected before any network call")

        runtime = _runtime_with_handler(handler)
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar"))
        with pytest.raises(FeatureNotSupported):
            await runtime.generate(request)

    async def test_metrics_hook_called_on_success_and_failure(self):
        calls = []

        class RecordingHook:
            def on_request(self, request, response, error, latency_ms):
                calls.append((response, error))

        def ok_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_completion_json())

        openai_client = AsyncOpenAI(
            base_url="http://localhost:8080/v1", api_key="x",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(ok_handler)),
            max_retries=0,
        )
        runtime = OpenAICompatibleRuntime(openai_client=openai_client, metrics_hook=RecordingHook())
        await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert len(calls) == 1
        assert calls[0][0] is not None and calls[0][1] is None


class TestOpenAICompatibleRuntimeStream:
    async def test_stream_yields_delta_text(self):
        sse_body = (
            'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
            'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
            'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
            '"choices":[{"index":0,"delta":{"content":", world"},"finish_reason":null}]}\n\n'
            'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=sse_body.encode(), headers={"content-type": "text/event-stream"}
            )

        runtime = _runtime_with_handler(handler)
        chunks = [c async for c in runtime.stream(LLMRequest(model="m", prompt="p"))]
        assert "".join(chunks) == "Hello, world"


class TestOpenAICompatibleRuntimeTokenize:
    async def test_tokenize_calls_native_endpoint_and_returns_ids(self):
        def chat_handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("tokenize must not call the chat completions endpoint")

        def tokenize_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/tokenize"
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})

        runtime = _runtime_with_handler(chat_handler, http_handler=tokenize_handler)
        tokens = await runtime.tokenize("m", "hello world")
        assert tokens == [1, 2, 3, 4]

    async def test_tokenize_strips_v1_suffix_from_base_url(self):
        captured_paths = []

        def tokenize_handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(str(request.url))
            return httpx.Response(200, json={"tokens": []})

        runtime = _runtime_with_handler(lambda r: httpx.Response(200), http_handler=tokenize_handler)
        await runtime.tokenize("m", "hi")
        assert captured_paths[0] == "http://localhost:8080/tokenize"

    async def test_tokenize_connection_error_raises_runtime_unavailable(self):
        def tokenize_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        runtime = _runtime_with_handler(lambda r: httpx.Response(200), http_handler=tokenize_handler)
        with pytest.raises(RuntimeUnavailable):
            await runtime.tokenize("m", "hi")

    async def test_tokenize_unexpected_shape_raises_invalid_model_response(self):
        def tokenize_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"not_tokens": "oops"})

        runtime = _runtime_with_handler(lambda r: httpx.Response(200), http_handler=tokenize_handler)
        with pytest.raises(InvalidModelResponse):
            await runtime.tokenize("m", "hi")
