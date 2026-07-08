"""Lab 8 — before/after latency on a repeated-query workload.

Unlike Labs 1-3, this needs NO live model runtime: FakeRuntime's simulated
latency (Module 6) produces a real, non-fabricated before/after number
right now, since the whole point of caching is avoiding *re-generation*,
which is equally true whether generation is real or simulated - the cache
doesn't know or care which.

Usage:
    uv run python scripts/module_06_5/lab_caching_before_after.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.gateway.cache import ResponseCache, response_cache_key  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402


def build_repeated_query_workload(unique_queries: list[str], repeats: int, model: str = "demo-model") -> list[LLMRequest]:
    """A workload with heavy repetition - the case caching exists for."""
    return [LLMRequest(model=model, prompt=query) for _ in range(repeats) for query in unique_queries]


async def run_workload_without_cache(runtime: FakeRuntime, requests: list[LLMRequest]) -> float:
    start = time.perf_counter()
    for request in requests:
        await runtime.generate(request)
    return time.perf_counter() - start


async def run_workload_with_cache(runtime: FakeRuntime, cache: ResponseCache, requests: list[LLMRequest]) -> float:
    start = time.perf_counter()
    for request in requests:
        key = response_cache_key(request.model, request.prompt, {"temperature": request.temperature}, "v1")
        if cache.get(key) is None:
            response = await runtime.generate(request)
            cache.put(key, response)
    return time.perf_counter() - start


async def run_lab(
    unique_query_count: int = 5, repeats: int = 4, simulated_latency_ms: float = 20.0
) -> dict[str, float]:
    unique_queries = [f"What is fact #{i}?" for i in range(unique_query_count)]
    workload = build_repeated_query_workload(unique_queries, repeats=repeats)

    uncached_runtime = FakeRuntime(default_response="a consistent answer", simulated_latency_ms=simulated_latency_ms)
    without_cache_seconds = await run_workload_without_cache(uncached_runtime, workload)

    cached_runtime = FakeRuntime(default_response="a consistent answer", simulated_latency_ms=simulated_latency_ms)
    cache = ResponseCache()
    with_cache_seconds = await run_workload_with_cache(cached_runtime, cache, workload)

    return {
        "workload_size": len(workload),
        "unique_queries": unique_query_count,
        "without_cache_seconds": without_cache_seconds,
        "with_cache_seconds": with_cache_seconds,
        "speedup": without_cache_seconds / with_cache_seconds if with_cache_seconds else 0.0,
        "cache_hit_rate": cache.hit_rate,
        "runtime_calls_without_cache": uncached_runtime.call_count,
        "runtime_calls_with_cache": cached_runtime.call_count,
    }


def results_to_markdown(results: dict[str, float]) -> str:
    return (
        "# Lab 8 — caching before/after\n\n"
        f"- Workload: {results['workload_size']:.0f} requests, {results['unique_queries']:.0f} unique queries\n"
        f"- Without cache: {results['without_cache_seconds']:.3f}s "
        f"({results['runtime_calls_without_cache']:.0f} runtime calls)\n"
        f"- With cache: {results['with_cache_seconds']:.3f}s "
        f"({results['runtime_calls_with_cache']:.0f} runtime calls)\n"
        f"- Speedup: {results['speedup']:.2f}x\n"
        f"- Cache hit rate: {results['cache_hit_rate']:.0%}\n"
    )


def main() -> int:
    results = asyncio.run(run_lab())
    print(results_to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
