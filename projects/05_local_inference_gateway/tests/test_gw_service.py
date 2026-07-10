from pathlib import Path

import pytest
from local_ai_core.deployment.config import AppConfig
from local_ai_core.gateway.admission_control import AdmissionController, AdmissionPolicy
from local_ai_core.gateway.queue import QueueFullError
from local_ai_core.optimization.fallback import NoRuntimesAvailable
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.tracing.trace import validate_trace_shape

from gw_router import TaskNotFoundError
from gw_service import build_gw_context, run_generate, run_gw_benchmark, run_stream

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"
ROUTES_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway_routes.yaml"


def make_config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "llama3.1:8b-instruct",
                "default_extraction": "qwen2.5:7b-instruct",
                "default_code": "qwen2.5-coder:7b",
                "default_embedding": "nomic-embed-text",
            },
        }
    )


class TestBuildGwContext:
    def test_wires_a_real_persistent_store_and_loads_real_routes(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH)

        assert set(ctx.routes) == {"extraction", "code", "chat"}
        assert (ctx.base.data_dir.base_dir / "gateway" / "gateway.db").exists()

    def test_fallback_runtime_defaults_to_the_base_runtime(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime()
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)
        assert ctx.fallback_runtime is runtime

    def test_accepts_an_independently_injected_fallback_runtime(self, tmp_path):
        config = make_config(tmp_path)
        primary_runtime = FakeRuntime()
        fallback_runtime = FakeRuntime()
        ctx = build_gw_context(
            config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH,
            runtime=primary_runtime, fallback_runtime=fallback_runtime,
        )
        assert ctx.fallback_runtime is fallback_runtime
        assert ctx.fallback_runtime is not ctx.base.runtime


@pytest.mark.asyncio
class TestRunGenerate:
    async def test_a_healthy_primary_answers_without_fallback(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "real chat answer"})
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        result = await run_generate(ctx, "chat", "hello")

        assert result.answer == "real chat answer"
        assert result.model_used == "llama3.1:8b-instruct"
        assert result.used_fallback is False

        logged = ctx.storage.list_requests(task="chat")
        assert logged[0].status == "ok"
        assert logged[0].trace_id == result.trace_id

    async def test_a_failing_primary_falls_through_to_the_fallback_model(self, tmp_path):
        config = make_config(tmp_path)
        primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
        fallback_runtime = FakeRuntime(responses={"qwen2.5:1.5b-instruct": "fallback chat answer"})
        ctx = build_gw_context(
            config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH,
            runtime=primary_runtime, fallback_runtime=fallback_runtime,
        )

        result = await run_generate(ctx, "chat", "hello")

        assert result.answer == "fallback chat answer"
        assert result.model_used == "qwen2.5:1.5b-instruct"
        assert result.used_fallback is True

    async def test_an_unknown_task_raises_before_any_model_call(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime()
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        with pytest.raises(TaskNotFoundError):
            await run_generate(ctx, "does-not-exist", "hello")
        assert runtime.call_count == 0

    async def test_both_models_failing_raises_and_is_logged(self, tmp_path):
        config = make_config(tmp_path)
        primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
        fallback_runtime = FakeRuntime(fail_with=RuntimeUnavailable("fallback down"))
        ctx = build_gw_context(
            config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH,
            runtime=primary_runtime, fallback_runtime=fallback_runtime,
        )

        with pytest.raises(NoRuntimesAvailable):
            await run_generate(ctx, "chat", "hello")

        logged = ctx.storage.list_requests(task="chat")
        assert logged[0].status == "no_runtimes_available"

    async def test_a_request_exceeding_timeout_raises_and_is_logged(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(simulated_latency_ms=50)
        ctx = build_gw_context(
            config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime,
            timeout_seconds=0.01,
        )

        with pytest.raises(RequestTimeout):
            await run_generate(ctx, "chat", "hello")

        logged = ctx.storage.list_requests(task="chat")
        assert logged[0].status == "timeout"

    async def test_a_full_queue_raises_queue_full_and_is_logged(self, tmp_path, monkeypatch):
        config = make_config(tmp_path)
        runtime = FakeRuntime()
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)
        ctx.base.admission_controller = AdmissionController(
            AdmissionPolicy(max_concurrent_requests=1, max_queue_size=0, reason="test: force queue_full")
        )

        async def _never_finishes(fn):
            raise QueueFullError(0, 0)

        monkeypatch.setattr(ctx.base.admission_controller, "submit", _never_finishes)

        with pytest.raises(QueueFullError):
            await run_generate(ctx, "chat", "hello")

        logged = ctx.storage.list_requests(task="chat")
        assert logged[0].status == "queue_full"

    async def test_every_request_gets_a_real_trace_id(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "answer"})
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        first = await run_generate(ctx, "chat", "hello")
        second = await run_generate(ctx, "chat", "hello again")

        assert first.trace_id != second.trace_id

    async def test_a_successful_request_produces_a_complete_real_trace(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "answer"})
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        result = await run_generate(ctx, "chat", "hello")

        missing = validate_trace_shape(result.trace)
        assert missing == []
        assert result.trace.total_elapsed_ms() > 0


@pytest.mark.asyncio
class TestRunStream:
    async def test_streams_from_the_primary_model(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "a b c"})
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        chunks = [chunk async for chunk in run_stream(ctx, "chat", "hello")]
        assert "".join(chunks).strip() == "a b c"

    async def test_an_unknown_task_raises_before_streaming(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime()
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        with pytest.raises(TaskNotFoundError):
            run_stream(ctx, "does-not-exist", "hello")


@pytest.mark.asyncio
class TestRunGwBenchmark:
    async def test_benchmarks_a_single_task(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "answer", "qwen2.5:1.5b-instruct": "answer"})
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        results = await run_gw_benchmark(ctx, task="chat", repeats=2)

        names = {r.name for r in results}
        assert names == {"chat-primary", "chat-fallback"}
        assert all(r.sample_count == 2 for r in results)

    async def test_benchmarks_every_task_when_none_specified(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime()
        ctx = build_gw_context(config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH, runtime=runtime)

        results = await run_gw_benchmark(ctx, repeats=1)

        names = {r.name for r in results}
        assert names == {
            "extraction-primary", "extraction-fallback", "code-primary", "code-fallback",
            "chat-primary", "chat-fallback",
        }
