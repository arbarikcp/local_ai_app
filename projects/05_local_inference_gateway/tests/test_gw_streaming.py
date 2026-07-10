import pytest
from local_ai_core.runtimes.errors import RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

from gw_model_binding import ModelBoundRuntime
from gw_streaming import stream_with_fallback


class MidStreamFailureRuntime:
    """A runtime whose stream() yields one real chunk, then raises - used
    to prove a failure AFTER the first chunk propagates as a stream error
    instead of silently retrying on a different model (ARCHITECTURE.md's
    documented constraint).
    """

    async def generate(self, request):
        raise NotImplementedError

    async def stream(self, request):
        yield "partial "
        raise RuntimeUnavailable("died mid-stream")

    async def tokenize(self, model, rendered_prompt):
        return []


@pytest.mark.asyncio
class TestStreamWithFallback:
    async def test_a_healthy_primary_streams_without_touching_fallback(self):
        primary_runtime = FakeRuntime(responses={"primary-model": "a b c"})
        fallback_runtime = FakeRuntime(responses={"fallback-model": "should not be used"})
        primary = ModelBoundRuntime(runtime=primary_runtime, model_id="primary-model")
        fallback = ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model")

        chunks = [
            chunk
            async for chunk in stream_with_fallback(LLMRequest(model="ignored", prompt="hi"), primary=primary, fallback=fallback)
        ]

        assert "".join(chunks).strip() == "a b c"
        assert fallback_runtime.call_count == 0

    async def test_a_pre_first_chunk_failure_falls_through_to_fallback(self):
        primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
        fallback_runtime = FakeRuntime(responses={"fallback-model": "recovered answer"})
        primary = ModelBoundRuntime(runtime=primary_runtime, model_id="primary-model")
        fallback = ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model")

        chunks = [
            chunk
            async for chunk in stream_with_fallback(LLMRequest(model="ignored", prompt="hi"), primary=primary, fallback=fallback)
        ]

        assert "".join(chunks).strip() == "recovered answer"

    async def test_a_post_first_chunk_failure_propagates_instead_of_retrying(self):
        primary = MidStreamFailureRuntime()
        fallback_runtime = FakeRuntime(responses={"fallback-model": "should not be reached"})
        fallback = ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model")

        chunks = []
        with pytest.raises(RuntimeUnavailable):
            async for chunk in stream_with_fallback(LLMRequest(model="ignored", prompt="hi"), primary=primary, fallback=fallback):
                chunks.append(chunk)

        assert chunks == ["partial "]
        assert fallback_runtime.call_count == 0

    async def test_an_empty_primary_stream_yields_nothing(self):
        class EmptyStreamRuntime:
            async def generate(self, request):
                raise NotImplementedError

            async def stream(self, request):
                return
                yield  # pragma: no cover - makes this an async generator

            async def tokenize(self, model, rendered_prompt):
                return []

        fallback_runtime = FakeRuntime(responses={"fallback-model": "should not be used"})
        fallback = ModelBoundRuntime(runtime=fallback_runtime, model_id="fallback-model")

        chunks = [
            chunk
            async for chunk in stream_with_fallback(
                LLMRequest(model="ignored", prompt="hi"), primary=EmptyStreamRuntime(), fallback=fallback
            )
        ]

        assert chunks == []
        assert fallback_runtime.call_count == 0
