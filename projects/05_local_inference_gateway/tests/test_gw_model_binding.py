import pytest
from local_ai_core.optimization.fallback import FallbackRuntime, NoRuntimesAvailable
from local_ai_core.runtimes.errors import RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

from gw_model_binding import ModelBoundRuntime


@pytest.mark.asyncio
class TestModelBoundRuntime:
    async def test_generate_rewrites_the_request_model(self):
        runtime = FakeRuntime(responses={"real-model": "answer from real model"})
        bound = ModelBoundRuntime(runtime=runtime, model_id="real-model")

        response = await bound.generate(LLMRequest(model="whatever-caller-said", prompt="hi"))

        assert response.text == "answer from real model"
        assert runtime.requests_received[0].model == "real-model"

    async def test_stream_rewrites_the_request_model(self):
        runtime = FakeRuntime(responses={"real-model": "a b c"})
        bound = ModelBoundRuntime(runtime=runtime, model_id="real-model")

        chunks = [chunk async for chunk in bound.stream(LLMRequest(model="whatever", prompt="hi"))]

        assert "".join(chunks).strip() == "a b c"
        assert runtime.requests_received[0].model == "real-model"

    async def test_tokenize_uses_the_bound_model_id(self):
        runtime = FakeRuntime()
        bound = ModelBoundRuntime(runtime=runtime, model_id="real-model")
        tokens = await bound.tokenize("ignored", "hello world")
        assert len(tokens) == 2


@pytest.mark.asyncio
class TestModelBoundRuntimeInsideFallbackRuntime:
    async def test_fallback_runtime_reuses_two_bound_models_unmodified(self):
        primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
        fallback_runtime = FakeRuntime(responses={"fallback-model": "answer from fallback"})

        chain = FallbackRuntime(
            [
                ModelBoundRuntime(runtime=primary_runtime, model_id="primary-model"),
                ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model"),
            ]
        )

        result = await chain.generate(LLMRequest(model="ignored", prompt="hi"))

        assert result.runtime_index == 1
        assert result.response.text == "answer from fallback"
        assert fallback_runtime.requests_received[0].model == "fallback-model"

    async def test_both_models_failing_raises_no_runtimes_available(self):
        primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
        fallback_runtime = FakeRuntime(fail_with=RuntimeUnavailable("fallback down"))

        chain = FallbackRuntime(
            [
                ModelBoundRuntime(runtime=primary_runtime, model_id="primary-model"),
                ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model"),
            ]
        )

        with pytest.raises(NoRuntimesAvailable):
            await chain.generate(LLMRequest(model="ignored", prompt="hi"))
