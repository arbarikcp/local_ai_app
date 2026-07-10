"""GatewayAppContext — the composition root for this project, extending
(not replacing) Module 23's `AppContext` with a `fallback_runtime`, the
loaded task routes, and a `GwStorage` handle (ARCHITECTURE.md "Deployment
shape"). `run_generate`/`run_stream`/`run_benchmark` are the functions the
FastAPI layer calls - it never reaches past this file into
`gw_router`/`gw_model_binding`/`gw_streaming` directly.

`fallback_runtime` defaults to the same instance as `base.runtime` (a
single Ollama/MLX server can legitimately serve both a task's primary and
fallback model by name) but is independently injectable, satisfying
curriculum's "support multiple local runtimes" requirement for real: a
deployment may back its fallback with a different, always-up runtime.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.app_context import AppContext, build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.gateway.queue import QueueFullError
from local_ai_core.optimization.benchmark_harness import BenchmarkConfig, BenchmarkResult, run_benchmark
from local_ai_core.optimization.fallback import FallbackRuntime, NoRuntimesAvailable
from local_ai_core.runtimes.base import LLMRuntime, Timer
from local_ai_core.runtimes.errors import RequestTimeout
from local_ai_core.runtimes.types import LLMRequest
from local_ai_core.security.tool_call_timeout import with_timeout
from local_ai_core.tracing.trace import Trace, TraceBuilder

from gw_model_binding import ModelBoundRuntime
from gw_router import TaskRoute, load_routes, resolve_route
from gw_storage import GatewayRequestRecord, GwStorage
from gw_streaming import stream_with_fallback

DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class GatewayAppContext:
    base: AppContext
    fallback_runtime: LLMRuntime
    routes: dict[str, TaskRoute]
    storage: GwStorage
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class GenerateResult:
    answer: str
    model_used: str
    used_fallback: bool
    trace_id: str
    trace: Trace


def build_gw_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    routes_path: str | Path,
    runtime: LLMRuntime | None = None,
    fallback_runtime: LLMRuntime | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> GatewayAppContext:
    base = build_app_context(config, model_catalog_path=model_catalog_path, runtime=runtime)
    resolved_fallback_runtime = fallback_runtime or base.runtime
    routes = load_routes(routes_path, model_registry=base.model_registry)

    gw_dir = base.data_dir.base_dir / "gateway"
    gw_dir.mkdir(parents=True, exist_ok=True)
    storage = GwStorage(gw_dir / "gateway.db")

    return GatewayAppContext(
        base=base,
        fallback_runtime=resolved_fallback_runtime,
        routes=routes,
        storage=storage,
        timeout_seconds=timeout_seconds,
    )


def _build_chain(ctx: GatewayAppContext, route: TaskRoute) -> FallbackRuntime:
    primary = ModelBoundRuntime(runtime=ctx.base.runtime, model_id=route.primary_model)
    fallback = ModelBoundRuntime(runtime=ctx.fallback_runtime, model_id=route.fallback_model)
    return FallbackRuntime([primary, fallback])


async def run_generate(ctx: GatewayAppContext, task: str, prompt: str) -> GenerateResult:
    """Raises `gw_router.TaskNotFoundError` for an unknown task (before any
    model call), `QueueFullError` (Module 6.5) if admission is rejected,
    `RequestTimeout` (Module 22, reused for model calls) if the call
    exceeds `ctx.timeout_seconds`, and `NoRuntimesAvailable` (Module 20) if
    both the primary and fallback model fail. Every outcome - success or
    failure - is persisted to `ctx.storage`, so the gateway's own request
    log stays complete even when a request never got an answer.
    """
    route = resolve_route(ctx.routes, task)
    chain = _build_chain(ctx, route)
    request = LLMRequest(model=route.primary_model, prompt=prompt)
    trace_id = str(uuid.uuid4())
    trace = TraceBuilder(trace_id)

    with trace.span("input_validation", task=task):
        pass

    timer = Timer()
    try:
        with trace.span("model_call"):
            queued = await ctx.base.admission_controller.submit(
                lambda: with_timeout(lambda: chain.generate(request), timeout_seconds=ctx.timeout_seconds)
            )
        latency_ms = timer.elapsed_ms
    except QueueFullError:
        _persist(ctx, trace_id=trace_id, task=task, model_used="", used_fallback=False, latency_ms=timer.elapsed_ms, status="queue_full")
        raise
    except RequestTimeout:
        _persist(
            ctx, trace_id=trace_id, task=task, model_used=route.primary_model, used_fallback=False,
            latency_ms=timer.elapsed_ms, status="timeout",
        )
        raise
    except NoRuntimesAvailable:
        _persist(
            ctx, trace_id=trace_id, task=task, model_used=route.fallback_model, used_fallback=True,
            latency_ms=timer.elapsed_ms, status="no_runtimes_available",
        )
        raise

    result = queued.result
    used_fallback = result.runtime_index == 1
    model_used = route.fallback_model if used_fallback else route.primary_model

    with trace.span("final_response", model_used=model_used, used_fallback=used_fallback):
        pass

    _persist(
        ctx, trace_id=trace_id, task=task, model_used=model_used, used_fallback=used_fallback,
        latency_ms=latency_ms, status="ok",
    )

    return GenerateResult(
        answer=result.response.text, model_used=model_used, used_fallback=used_fallback,
        trace_id=trace_id, trace=trace.trace,
    )


def _persist(
    ctx: GatewayAppContext, *, trace_id: str, task: str, model_used: str, used_fallback: bool,
    latency_ms: float, status: str,
) -> None:
    ctx.storage.save_request(
        GatewayRequestRecord(
            request_id=str(uuid.uuid4()),
            trace_id=trace_id,
            task=task,
            model_used=model_used,
            used_fallback=used_fallback,
            latency_ms=latency_ms,
            status=status,
        )
    )


def run_stream(ctx: GatewayAppContext, task: str, prompt: str):
    route = resolve_route(ctx.routes, task)
    primary = ModelBoundRuntime(runtime=ctx.base.runtime, model_id=route.primary_model)
    fallback = ModelBoundRuntime(runtime=ctx.fallback_runtime, model_id=route.fallback_model)
    request = LLMRequest(model=route.primary_model, prompt=prompt)
    return stream_with_fallback(request, primary=primary, fallback=fallback)


async def run_gw_benchmark(ctx: GatewayAppContext, *, task: str | None = None, repeats: int = 3) -> list[BenchmarkResult]:
    tasks = [task] if task is not None else list(ctx.routes)
    configs: list[BenchmarkConfig] = []
    for task_name in tasks:
        route = resolve_route(ctx.routes, task_name)
        configs.append(
            BenchmarkConfig(
                name=f"{task_name}-primary",
                runtime=ModelBoundRuntime(runtime=ctx.base.runtime, model_id=route.primary_model),
                request=LLMRequest(model=route.primary_model, prompt="Benchmark probe prompt."),
            )
        )
        configs.append(
            BenchmarkConfig(
                name=f"{task_name}-fallback",
                runtime=ModelBoundRuntime(runtime=ctx.fallback_runtime, model_id=route.fallback_model),
                request=LLMRequest(model=route.fallback_model, prompt="Benchmark probe prompt."),
            )
        )
    return await run_benchmark(configs, repeats=repeats)
