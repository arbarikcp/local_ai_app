"""Labs 1-3 — measure queue wait, latency, and failure rate at 1/2/4
concurrent requests, using this module's own BoundedRequestQueue /
AdmissionController (Module 6.5) instead of Module 4's raw concurrency
simulation, which had no queueing/admission layer.

Usage:
    uv run python scripts/module_06_5/lab_measure_concurrency.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from local_ai_core.gateway.admission_control import (  # noqa: E402
    AdmissionController,
    AdmissionPolicy,
    ConcurrencyMeasurement,
    recommend_policy_from_measurements,
)
from local_ai_core.runtimes.errors import LLMError  # noqa: E402
from local_ai_core.runtimes.ollama import OllamaRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from ollama_probe import is_ollama_available  # noqa: E402

DEFAULT_LEVELS = [1, 2, 4]
DEFAULT_PROMPT = "Explain what a KV cache is in one sentence."
DEFAULT_REQUESTS_PER_LEVEL = 8


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile - simple and adequate for small lab sample sizes."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct * (len(ordered) - 1)))
    return ordered[idx]


async def measure_concurrency_level(
    runtime, model: str, prompt: str, n_concurrent: int, n_requests: int
) -> ConcurrencyMeasurement:
    policy = AdmissionPolicy(
        max_concurrent_requests=n_concurrent,
        max_queue_size=max(n_requests, 1),
        reason=f"lab_measure_concurrency.py run at concurrency={n_concurrent}",
    )
    controller = AdmissionController(policy)

    latencies: list[float] = []
    failures = 0

    async def run_one() -> None:
        nonlocal failures

        async def call():
            return await runtime.generate(LLMRequest(model=model, prompt=prompt))

        try:
            result = await controller.submit(call)
            latencies.append(result.queue_wait_seconds + result.execution_seconds)
        except LLMError:
            failures += 1

    await asyncio.gather(*[run_one() for _ in range(n_requests)])

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return ConcurrencyMeasurement(
        concurrency=n_concurrent,
        mean_latency_seconds=mean_latency,
        p95_latency_seconds=percentile(latencies, 0.95),
        failure_rate=failures / n_requests,
    )


def measurements_to_markdown_table(measurements: list[ConcurrencyMeasurement]) -> str:
    header = "| Concurrency | Mean latency (s) | p95 latency (s) | Failure rate |\n|---:|---:|---:|---:|\n"
    lines = [
        f"| {m.concurrency} | {m.mean_latency_seconds:.2f} | {m.p95_latency_seconds:.2f} | {m.failure_rate:.0%} |"
        for m in measurements
    ]
    return header + "\n".join(lines)


async def run_lab(model: str, prompt: str, levels: list[int], n_requests: int) -> list[ConcurrencyMeasurement]:
    runtime = OllamaRuntime()
    try:
        return [await measure_concurrency_level(runtime, model, prompt, n, n_requests) for n in levels]
    finally:
        await runtime.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--requests-per-level", type=int, default=DEFAULT_REQUESTS_PER_LEVEL)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac.",
            file=sys.stderr,
        )
        return 1

    measurements = asyncio.run(run_lab(args.model, args.prompt, args.levels, args.requests_per_level))

    print(f"# Labs 1-3 — concurrency measurement\n\nModel: `{args.model}`\n")
    print(measurements_to_markdown_table(measurements))

    recommended = recommend_policy_from_measurements(measurements)
    print(f"\nRecommended policy: max_concurrent_requests={recommended.max_concurrent_requests}")
    print(f"Reason: {recommended.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
