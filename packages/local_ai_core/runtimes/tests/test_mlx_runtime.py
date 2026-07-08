from dataclasses import dataclass
from typing import Any

import pytest

from local_ai_core.runtimes.errors import FeatureNotSupported, InvalidModelResponse, ModelNotLoaded
from local_ai_core.runtimes.mlx import MLXRuntime, render_prompt
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat


class FakeTokenizer:
    """Duck-types mlx_lm's tokenizer surface: apply_chat_template + encode."""

    def __init__(self, *, has_chat_template: bool = True) -> None:
        self.has_chat_template = has_chat_template

    def apply_chat_template(self, messages: list[dict], tokenize: bool, add_generation_prompt: bool) -> str:
        if not self.has_chat_template:
            raise AttributeError("no chat template configured")
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + "<assistant>"

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:5]]  # fake, deterministic, cheap


class TokenizerWithoutChatTemplate:
    """A tokenizer object with no apply_chat_template attribute at all."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


@dataclass(frozen=True)
class FakeStreamItem:
    text: str


def fake_load_fn(model_id: str) -> tuple[Any, Any]:
    return (f"model:{model_id}", FakeTokenizer())


def fake_generate_fn(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> str:
    return f"response to: {prompt}"


def fake_stream_generate_fn(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int):
    for word in ["Hello", ", ", "world", "!"]:
        yield FakeStreamItem(text=word)


# --- render_prompt (pure) ---------------------------------------------------


class TestRenderPrompt:
    def test_uses_chat_template_when_available(self):
        tokenizer = FakeTokenizer()
        req = LLMRequest(model="m", prompt="hi", system="Be terse.")
        rendered = render_prompt(tokenizer, req)
        assert rendered == "<system>Be terse.<user>hi<assistant>"

    def test_no_system_message_when_absent(self):
        tokenizer = FakeTokenizer()
        req = LLMRequest(model="m", prompt="hi")
        rendered = render_prompt(tokenizer, req)
        assert rendered == "<user>hi<assistant>"

    def test_falls_back_to_concatenation_without_chat_template(self):
        tokenizer = TokenizerWithoutChatTemplate()
        req = LLMRequest(model="m", prompt="hi", system="Be terse.")
        rendered = render_prompt(tokenizer, req)
        assert rendered == "Be terse.\n\nhi"

    def test_falls_back_to_bare_prompt_without_system_or_chat_template(self):
        tokenizer = TokenizerWithoutChatTemplate()
        req = LLMRequest(model="m", prompt="hi")
        assert render_prompt(tokenizer, req) == "hi"


# --- MLXRuntime.generate -----------------------------------------------------


class TestMLXRuntimeGenerate:
    async def test_successful_generate_returns_llm_response(self):
        runtime = MLXRuntime(load_fn=fake_load_fn, generate_fn=fake_generate_fn)
        response = await runtime.generate(LLMRequest(model="test-model", prompt="hi"))
        assert "response to:" in response.text
        assert response.model == "test-model"
        assert response.prompt_tokens is not None
        assert response.completion_tokens is not None

    async def test_caches_loaded_model_across_calls(self):
        load_calls = []

        def counting_load_fn(model_id: str):
            load_calls.append(model_id)
            return fake_load_fn(model_id)

        runtime = MLXRuntime(load_fn=counting_load_fn, generate_fn=fake_generate_fn)
        await runtime.generate(LLMRequest(model="m", prompt="p1"))
        await runtime.generate(LLMRequest(model="m", prompt="p2"))
        assert load_calls == ["m"]  # loaded once, reused (theory doc §6 warmup/lifecycle)

    async def test_load_failure_raises_model_not_loaded(self):
        def failing_load_fn(model_id: str):
            raise OSError("model file not found")

        runtime = MLXRuntime(load_fn=failing_load_fn, generate_fn=fake_generate_fn)
        with pytest.raises(ModelNotLoaded):
            await runtime.generate(LLMRequest(model="missing-model", prompt="p"))

    async def test_generate_failure_raises_invalid_model_response(self):
        def failing_generate_fn(model, tokenizer, *, prompt, max_tokens):
            raise RuntimeError("out of memory during generation")

        runtime = MLXRuntime(load_fn=fake_load_fn, generate_fn=failing_generate_fn)
        with pytest.raises(InvalidModelResponse):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_json_schema_format_raises_feature_not_supported_before_loading(self):
        def unreachable_load_fn(model_id: str):
            pytest.fail("must be rejected before any load attempt")

        runtime = MLXRuntime(load_fn=unreachable_load_fn, generate_fn=fake_generate_fn)
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="json_schema"))
        with pytest.raises(FeatureNotSupported):
            await runtime.generate(request)

    async def test_grammar_format_raises_feature_not_supported(self):
        runtime = MLXRuntime(load_fn=fake_load_fn, generate_fn=fake_generate_fn)
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="x"))
        with pytest.raises(FeatureNotSupported):
            await runtime.generate(request)

    async def test_metrics_hook_called_on_success(self):
        calls = []

        class RecordingHook:
            def on_request(self, request, response, error, latency_ms):
                calls.append((response, error))

        runtime = MLXRuntime(load_fn=fake_load_fn, generate_fn=fake_generate_fn, metrics_hook=RecordingHook())
        await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert len(calls) == 1
        assert calls[0][0] is not None
        assert calls[0][1] is None

    async def test_token_counts_are_none_when_tokenizer_has_no_encode(self):
        class NoEncodeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

        def load_fn(model_id):
            return ("model", NoEncodeTokenizer())

        runtime = MLXRuntime(load_fn=load_fn, generate_fn=fake_generate_fn)
        response = await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert response.prompt_tokens is None
        assert response.completion_tokens is None


# --- MLXRuntime.stream --------------------------------------------------------


class TestMLXRuntimeStream:
    async def test_stream_yields_text_incrementally(self):
        runtime = MLXRuntime(
            load_fn=fake_load_fn, generate_fn=fake_generate_fn, stream_generate_fn=fake_stream_generate_fn
        )
        chunks = [c async for c in runtime.stream(LLMRequest(model="m", prompt="p"))]
        assert chunks == ["Hello", ", ", "world", "!"]

    async def test_stream_propagates_generator_failures(self):
        def failing_stream_fn(model, tokenizer, *, prompt, max_tokens):
            yield FakeStreamItem(text="partial")
            raise RuntimeError("stream broke")

        runtime = MLXRuntime(
            load_fn=fake_load_fn, generate_fn=fake_generate_fn, stream_generate_fn=failing_stream_fn
        )
        received = []
        with pytest.raises(InvalidModelResponse):
            async for chunk in runtime.stream(LLMRequest(model="m", prompt="p")):
                received.append(chunk)
        assert received == ["partial"]  # got the good chunk before the failure

    async def test_stream_rejects_unsupported_response_format_before_loading(self):
        def unreachable_load_fn(model_id: str):
            pytest.fail("must be rejected before any load attempt")

        runtime = MLXRuntime(load_fn=unreachable_load_fn, generate_fn=fake_generate_fn)
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="x"))
        with pytest.raises(FeatureNotSupported):
            async for _ in runtime.stream(request):
                pass


# --- MLXRuntime.tokenize -------------------------------------------------------


class TestMLXRuntimeTokenize:
    async def test_returns_real_token_ids_from_the_tokenizer(self):
        runtime = MLXRuntime(load_fn=fake_load_fn, generate_fn=fake_generate_fn)
        tokens = await runtime.tokenize("m", "hello")
        assert tokens == [ord(c) for c in "hello"]

    async def test_reuses_cached_model_for_tokenize(self):
        load_calls = []

        def counting_load_fn(model_id: str):
            load_calls.append(model_id)
            return fake_load_fn(model_id)

        runtime = MLXRuntime(load_fn=counting_load_fn, generate_fn=fake_generate_fn)
        await runtime.generate(LLMRequest(model="m", prompt="p"))
        await runtime.tokenize("m", "another prompt")
        assert load_calls == ["m"]

    async def test_tokenize_failure_raises_invalid_model_response(self):
        class BrokenTokenizer:
            def encode(self, text):
                raise ValueError("bad input")

        def load_fn(model_id):
            return ("model", BrokenTokenizer())

        runtime = MLXRuntime(load_fn=load_fn, generate_fn=fake_generate_fn)
        with pytest.raises(InvalidModelResponse):
            await runtime.tokenize("m", "hi")
