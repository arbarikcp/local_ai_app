"""Benchmark harness (theory doc Lab 1) — real latency and tokens-per-second
measurement across named runtime configurations. `FakeRuntime`'s
`simulated_latency_ms` (Module 6.5 precedent) makes the timing numbers
genuine, real elapsed time, not asserted constants - only the *runtime
being fast/slow for a real reason* (an actual quantization or model choice)
stays honest-skip on this machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    runtime: LLMRuntime
    request: LLMRequest


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    sample_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile * (len(ordered) - 1)))
    return ordered[index]


def _tokens_per_second(completion_tokens: int | None, latency_ms: float | None) -> float:
    if not completion_tokens or not latency_ms or latency_ms <= 0:
        return 0.0
    return completion_tokens / (latency_ms / 1000)


async def run_benchmark(configs: list[BenchmarkConfig], *, repeats: int = 3) -> list[BenchmarkResult]:
    """Runs each config's runtime `repeats` times, sequentially, and
    aggregates real per-call latency (`response.latency_ms`) and derived
    tokens-per-second into one summary row per config.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    results: list[BenchmarkResult] = []
    for config in configs:
        latencies: list[float] = []
        tokens_per_second_samples: list[float] = []
        for _ in range(repeats):
            response = await config.runtime.generate(config.request)
            latency_ms = response.latency_ms or 0.0
            latencies.append(latency_ms)
            tokens_per_second_samples.append(_tokens_per_second(response.completion_tokens, latency_ms))

        results.append(
            BenchmarkResult(
                name=config.name,
                sample_count=repeats,
                mean_latency_ms=sum(latencies) / len(latencies),
                p95_latency_ms=_percentile(latencies, 0.95),
                mean_tokens_per_second=sum(tokens_per_second_samples) / len(tokens_per_second_samples),
            )
        )
    return results
