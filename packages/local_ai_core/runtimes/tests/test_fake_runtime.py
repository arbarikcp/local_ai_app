import pytest

from local_ai_core.runtimes.base import with_retries
from local_ai_core.runtimes.errors import FeatureNotSupported, RequestTimeout, RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat


class TestGenerate:
    async def test_returns_default_response_for_unlisted_model(self):
        runtime = FakeRuntime(default_response="hello there")
        response = await runtime.generate(LLMRequest(model="unlisted-model", prompt="hi"))
        assert response.text == "hello there"
        assert response.model == "unlisted-model"

    async def test_returns_per_model_response(self):
        runtime = FakeRuntime(responses={"model-a": "response A", "model-b": "response B"})
        resp_a = await runtime.generate(LLMRequest(model="model-a", prompt="hi"))
        resp_b = await runtime.generate(LLMRequest(model="model-b", prompt="hi"))
        assert resp_a.text == "response A"
        assert resp_b.text == "response B"

    async def test_computes_prompt_and_completion_token_counts(self):
        runtime = FakeRuntime(default_response="one two three")
        response = await runtime.generate(LLMRequest(model="m", prompt="a b c d"))
        assert response.prompt_tokens == 4
        assert response.completion_tokens == 3

    async def test_records_latency(self):
        runtime = FakeRuntime()
        response = await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert response.latency_ms is not None
        assert response.latency_ms >= 0

    async def test_tracks_call_count_and_request_history(self):
        runtime = FakeRuntime()
        await runtime.generate(LLMRequest(model="m", prompt="first"))
        await runtime.generate(LLMRequest(model="m", prompt="second"))
        assert runtime.call_count == 2
        assert [r.prompt for r in runtime.requests_received] == ["first", "second"]

    async def test_fail_with_raises_the_configured_error_every_time(self):
        runtime = FakeRuntime(fail_with=RuntimeUnavailable("simulated down"))
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_grammar_response_format_raises_feature_not_supported(self):
        runtime = FakeRuntime()
        request = LLMRequest(model="m", prompt="p", response_format=ResponseFormat(type="grammar", grammar="root ::= 'x'"))
        with pytest.raises(FeatureNotSupported):
            await runtime.generate(request)

    async def test_text_and_json_schema_response_formats_succeed(self):
        runtime = FakeRuntime()
        for fmt in [ResponseFormat(type="text"), ResponseFormat(type="json_schema", schema={"type": "object"})]:
            request = LLMRequest(model="m", prompt="p", response_format=fmt)
            response = await runtime.generate(request)
            assert response.text


class TestFailFirstNCalls:
    async def test_fails_exactly_n_times_then_succeeds(self):
        runtime = FakeRuntime(fail_first_n_calls=2, default_response="ok")
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))
        with pytest.raises(RuntimeUnavailable):
            await runtime.generate(LLMRequest(model="m", prompt="p"))
        response = await runtime.generate(LLMRequest(model="m", prompt="p"))
        assert response.text == "ok"

    async def test_uses_custom_transient_error_type(self):
        runtime = FakeRuntime(fail_first_n_calls=1, transient_error_for_first_n=RequestTimeout("simulated timeout"))
        with pytest.raises(RequestTimeout):
            await runtime.generate(LLMRequest(model="m", prompt="p"))

    async def test_composes_correctly_with_with_retries(self):
        # This is the whole point of fail_first_n_calls: proving the retry
        # wrapper actually retries against a runtime that fails predictably.
        runtime = FakeRuntime(fail_first_n_calls=2, default_response="succeeded after retries")

        async def call():
            return await runtime.generate(LLMRequest(model="m", prompt="p"))

        response = await with_retries(call, max_attempts=5, sleep_fn=_no_sleep)
        assert response.text == "succeeded after retries"
        assert runtime.call_count == 3


class TestStream:
    async def test_yields_the_response_text_word_by_word(self):
        runtime = FakeRuntime(default_response="one two three")
        chunks = [chunk async for chunk in runtime.stream(LLMRequest(model="m", prompt="p"))]
        assert "".join(chunks).strip() == "one two three"
        assert len(chunks) == 3

    async def test_stream_raises_configured_error(self):
        runtime = FakeRuntime(fail_with=RuntimeUnavailable("down"))
        with pytest.raises(RuntimeUnavailable):
            async for _ in runtime.stream(LLMRequest(model="m", prompt="p")):
                pass

    async def test_stream_tracks_call_count(self):
        runtime = FakeRuntime(default_response="a b")
        async for _ in runtime.stream(LLMRequest(model="m", prompt="p")):
            pass
        assert runtime.call_count == 1


class TestTokenize:
    async def test_returns_one_fake_id_per_word(self):
        runtime = FakeRuntime()
        ids = await runtime.tokenize("m", "the quick brown fox")
        assert len(ids) == 4
        assert all(isinstance(i, int) for i in ids)

    async def test_is_deterministic_for_the_same_input(self):
        runtime = FakeRuntime()
        ids1 = await runtime.tokenize("m", "hello world")
        ids2 = await runtime.tokenize("m", "hello world")
        assert ids1 == ids2

    async def test_empty_prompt_returns_empty_list(self):
        runtime = FakeRuntime()
        assert await runtime.tokenize("m", "") == []


async def _no_sleep(seconds: float) -> None:
    return None
