"""Lab 1 - build a benchmark harness and use it to compare three simulated
"configurations" - fast, medium, slow. Real latency measurement via
`FakeRuntime`'s `simulated_latency_ms` (Module 6.5 precedent): the numbers
below are genuinely computed elapsed time, not asserted constants - only
the *reason* one real runtime is faster than another (an actual
quantization/model choice) stays honest-skip on this machine.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.optimization.benchmark_harness import BenchmarkConfig, run_benchmark  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

REQUEST = LLMRequest(model="ticket-classifier", prompt="Classify this support ticket.")


async def run_lab() -> list:
    configs = [
        BenchmarkConfig(
            name="q4_small_model",
            runtime=FakeRuntime(default_response="billing", simulated_latency_ms=5),
            request=REQUEST,
        ),
        BenchmarkConfig(
            name="q8_medium_model",
            runtime=FakeRuntime(default_response="billing category", simulated_latency_ms=20),
            request=REQUEST,
        ),
        BenchmarkConfig(
            name="fp16_large_model",
            runtime=FakeRuntime(default_response="this is a billing category ticket", simulated_latency_ms=60),
            request=REQUEST,
        ),
    ]
    return await run_benchmark(configs, repeats=5)


def results_to_markdown(results: list) -> str:
    lines = [
        "# Lab 1 - benchmark harness",
        "",
        "| Config | Samples | Mean latency (ms) | p95 latency (ms) | Mean tokens/sec |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.sample_count} | {r.mean_latency_ms:.2f} | {r.p95_latency_ms:.2f} | "
            f"{r.mean_tokens_per_second:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    results = asyncio.run(run_lab())
    print(results_to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
