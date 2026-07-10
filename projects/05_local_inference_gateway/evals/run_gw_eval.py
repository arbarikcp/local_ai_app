"""Evaluation command against curriculum's own 10 functional requirements
(PROPOSAL.md "How success is measured"). Unlike Projects 1/2/4's
accuracy-against-a-labeled-dataset evals, a gateway has no "correct
answer" to score — its job is correct *behavior* under real conditions
(routing, fallback, timeouts, concurrency, streaming). So this harness
runs one real, scripted scenario per requirement and proves the real
outcome matches what was claimed, the same "prove it, don't assert it"
discipline as every other project's eval, applied to reliability behavior
instead of answer quality.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))
sys.path.insert(0, str(_PROJECT_ROOT / "schemas"))

from local_ai_core.deployment.config import AppConfig  # noqa: E402
from local_ai_core.gateway.admission_control import AdmissionController, AdmissionPolicy  # noqa: E402
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.optimization.fallback import NoRuntimesAvailable  # noqa: E402
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.tracing.trace import validate_trace_shape  # noqa: E402

from gw_router import TaskNotFoundError  # noqa: E402
from gw_service import build_gw_context, run_generate, run_gw_benchmark, run_stream  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"
ROUTES_PATH = _PROJECT_ROOT / "config" / "gateway_routes.yaml"


@dataclass(frozen=True)
class ScenarioResult:
    requirement: str
    scenario: str
    passed: bool
    detail: str


def _config(tmp_dir: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": tmp_dir},
            "models": {
                "default_chat": "llama3.1:8b-instruct",
                "default_extraction": "qwen2.5:7b-instruct",
                "default_code": "qwen2.5-coder:7b",
                "default_embedding": "nomic-embed-text",
            },
        }
    )


async def _scenario_multiple_runtimes(tmp_dir: str) -> ScenarioResult:
    ollama_shaped_runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "answer from runtime A"})
    mlx_shaped_runtime = FakeRuntime(responses={"qwen2.5:1.5b-instruct": "answer from runtime B"})
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH,
        runtime=ollama_shaped_runtime, fallback_runtime=mlx_shaped_runtime,
    )
    result = await run_generate(ctx, "chat", "hello")
    passed = result.answer == "answer from runtime A" and ollama_shaped_runtime.call_count == 1
    return ScenarioResult(
        "1. Multiple local runtimes", "two independently injected FakeRuntime instances back primary/fallback",
        passed, f"served by runtime A, answer={result.answer!r}",
    )


async def _scenario_model_registry(tmp_dir: str) -> ScenarioResult:
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=FakeRuntime(),
    )
    entry = ctx.base.model_registry.get("llama3.1:8b-instruct")
    passed = entry is not None and len(ctx.base.model_registry) == 10
    return ScenarioResult(
        "2. Model registry", "routes.yaml validated against the real 10-entry MODEL_CATALOG.md",
        passed, f"registry has {len(ctx.base.model_registry)} entries",
    )


async def _scenario_task_routing(tmp_dir: str) -> ScenarioResult:
    runtime = FakeRuntime(responses={"qwen2.5-coder:7b": "code answer"})
    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=runtime)
    result = await run_generate(ctx, "code", "write a function")
    passed = result.model_used == "qwen2.5-coder:7b"
    return ScenarioResult(
        "3. Task-based routing", "task='code' is served by code's configured primary model",
        passed, f"model_used={result.model_used!r}",
    )


async def _scenario_streaming(tmp_dir: str) -> ScenarioResult:
    runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "one two three"})
    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=runtime)
    chunks = [chunk async for chunk in run_stream(ctx, "chat", "hello")]
    passed = len(chunks) > 1 and "".join(chunks).strip() == "one two three"
    return ScenarioResult(
        "4. Streaming", "a real chunk-by-chunk delivery, not one buffered blob",
        passed, f"received {len(chunks)} chunks",
    )


async def _scenario_fallback_model(tmp_dir: str) -> ScenarioResult:
    primary_runtime = FakeRuntime(fail_with=RuntimeUnavailable("primary down"))
    fallback_runtime = FakeRuntime(responses={"qwen2.5:1.5b-instruct": "fallback answer"})
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH,
        runtime=primary_runtime, fallback_runtime=fallback_runtime,
    )
    result = await run_generate(ctx, "chat", "hello")
    passed = result.used_fallback is True and result.model_used == "qwen2.5:1.5b-instruct"
    return ScenarioResult(
        "6. Fallback model", "a failing primary falls through to the configured fallback model",
        passed, f"used_fallback={result.used_fallback}, model_used={result.model_used!r}",
    )


async def _scenario_timeouts(tmp_dir: str) -> ScenarioResult:
    runtime = FakeRuntime(simulated_latency_ms=50)
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=runtime,
        timeout_seconds=0.01,
    )
    try:
        await run_generate(ctx, "chat", "hello")
        passed, detail = False, "no RequestTimeout was raised"
    except RequestTimeout:
        logged = ctx.storage.list_requests(task="chat")
        passed = logged[0].status == "timeout"
        detail = f"RequestTimeout raised, logged status={logged[0].status!r}"
    return ScenarioResult("5. Timeouts", "a 50ms call against a 10ms timeout raises and is logged", passed, detail)


async def _scenario_concurrency_limit(tmp_dir: str) -> ScenarioResult:
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH,
        runtime=FakeRuntime(simulated_latency_ms=50),
    )
    ctx.base.admission_controller = AdmissionController(
        AdmissionPolicy(max_concurrent_requests=1, max_queue_size=0, reason="eval: force queue_full deterministically")
    )
    results = await asyncio.gather(
        run_generate(ctx, "chat", "first"), run_generate(ctx, "chat", "second"), return_exceptions=True
    )
    passed = any(isinstance(r, QueueFullError) for r in results) and any(not isinstance(r, Exception) for r in results)
    return ScenarioResult(
        "7. Concurrency limit", "max_queue_size=0 rejects a second concurrent request with QueueFullError",
        passed, f"outcomes: {[type(r).__name__ if isinstance(r, Exception) else 'ok' for r in results]}",
    )


async def _scenario_trace_logging(tmp_dir: str) -> ScenarioResult:
    runtime = FakeRuntime(responses={"llama3.1:8b-instruct": "answer"})
    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=runtime)
    result = await run_generate(ctx, "chat", "hello")
    missing = validate_trace_shape(result.trace)
    logged = ctx.storage.get_request(ctx.storage.list_requests(task="chat")[0].request_id)
    passed = missing == [] and logged is not None and logged.trace_id == result.trace_id
    return ScenarioResult(
        "8. Trace logging", "every request produces a real span tree with all core steps, persisted by trace_id",
        passed, f"missing_core_steps={missing}, span_names={result.trace.span_names()}",
    )


async def _scenario_benchmark(tmp_dir: str) -> ScenarioResult:
    runtime = FakeRuntime(simulated_latency_ms=5, responses={"llama3.1:8b-instruct": "a b c d"})
    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=runtime)
    results = await run_gw_benchmark(ctx, task="chat", repeats=3)
    passed = len(results) == 2 and all(r.sample_count == 3 and r.mean_latency_ms > 0 for r in results)
    return ScenarioResult(
        "9. Benchmark", "run_gw_benchmark() returns real, freshly-measured latency for chat's primary+fallback",
        passed, f"{[(r.name, round(r.mean_latency_ms, 2)) for r in results]}",
    )


async def _scenario_health_checks(tmp_dir: str) -> ScenarioResult:
    from local_ai_core.deployment.health import run_liveness_check, run_readiness_check

    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=FakeRuntime())
    liveness = run_liveness_check()
    readiness = run_readiness_check(ctx.base.data_dir, ctx.base.model_registry)
    passed = liveness.passed and readiness.passed
    return ScenarioResult(
        "10. Health checks", "run_liveness_check()/run_readiness_check() both pass against the real gateway context",
        passed, f"liveness={liveness.passed}, readiness={readiness.passed}",
    )


async def _scenario_unknown_task_is_rejected(tmp_dir: str) -> ScenarioResult:
    ctx = build_gw_context(_config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH, runtime=FakeRuntime())
    try:
        await run_generate(ctx, "does-not-exist", "hello")
        passed, detail = False, "no TaskNotFoundError was raised"
    except TaskNotFoundError:
        passed, detail = True, "TaskNotFoundError raised before any model call"
    return ScenarioResult(
        "3b. Unknown task rejected", "a task with no configured route is rejected, not silently misrouted",
        passed, detail,
    )


async def _scenario_both_models_failing(tmp_dir: str) -> ScenarioResult:
    ctx = build_gw_context(
        _config(tmp_dir), model_catalog_path=CATALOG_PATH, routes_path=ROUTES_PATH,
        runtime=FakeRuntime(fail_with=RuntimeUnavailable("primary down")),
        fallback_runtime=FakeRuntime(fail_with=RuntimeUnavailable("fallback down")),
    )
    try:
        await run_generate(ctx, "chat", "hello")
        passed, detail = False, "no NoRuntimesAvailable was raised"
    except NoRuntimesAvailable:
        logged = ctx.storage.list_requests(task="chat")
        passed = logged[0].status == "no_runtimes_available"
        detail = f"NoRuntimesAvailable raised, logged status={logged[0].status!r}"
    return ScenarioResult(
        "6b. Both models failing", "a failing primary AND fallback raises NoRuntimesAvailable and is logged",
        passed, detail,
    )


SCENARIOS = [
    _scenario_multiple_runtimes,
    _scenario_model_registry,
    _scenario_task_routing,
    _scenario_unknown_task_is_rejected,
    _scenario_streaming,
    _scenario_fallback_model,
    _scenario_both_models_failing,
    _scenario_timeouts,
    _scenario_concurrency_limit,
    _scenario_trace_logging,
    _scenario_benchmark,
    _scenario_health_checks,
]


async def run_eval() -> list[ScenarioResult]:
    results = []
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory(prefix="project05-eval-") as tmp_dir:
            results.append(await scenario(tmp_dir))
    return results


def results_to_markdown(results: list[ScenarioResult]) -> str:
    lines = ["# Evaluation against curriculum's 10 gateway functional requirements", ""]
    passed_count = sum(1 for r in results if r.passed)
    lines.append(f"**{passed_count}/{len(results)} scenarios passed.**")
    lines.append("")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"- [{mark}] {r.requirement} — {r.scenario}")
        lines.append(f"  - {r.detail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    results = asyncio.run(run_eval())
    print(results_to_markdown(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
