"""Lab 7 - add a performance dashboard. Drives a mix of successful and
failing requests through `InMemoryMetricsHook`, then prints the real
aggregated `PerformanceDashboard` summary (p50/p95 latency, error rate,
mean tokens/sec).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.optimization.dashboard import InMemoryMetricsHook, PerformanceDashboard  # noqa: E402
from local_ai_core.runtimes.errors import RuntimeUnavailable  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

REQUEST = LLMRequest(model="ticket-classifier", prompt="Classify this ticket.")


async def run_lab() -> dict:
    hook = InMemoryMetricsHook()
    healthy_runtime = FakeRuntime(default_response="billing category", simulated_latency_ms=8, metrics_hook=hook)
    flaky_runtime = FakeRuntime(fail_with=RuntimeUnavailable("overloaded"), metrics_hook=hook)
    dashboard = PerformanceDashboard(hook=hook)

    for _ in range(8):
        await healthy_runtime.generate(REQUEST)
    for _ in range(2):
        try:
            await flaky_runtime.generate(REQUEST)
        except RuntimeUnavailable:
            pass

    summary = dashboard.summary()
    return {
        "request_count": summary.request_count,
        "error_count": summary.error_count,
        "error_rate": summary.error_rate,
        "mean_latency_ms": summary.mean_latency_ms,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "mean_tokens_per_second": summary.mean_tokens_per_second,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 7 - performance dashboard\n\n"
        f"- Requests: {result['request_count']} (errors: {result['error_count']}, "
        f"error rate: {result['error_rate']:.0%})\n"
        f"- Mean latency: {result['mean_latency_ms']:.2f}ms\n"
        f"- p50 latency: {result['p50_latency_ms']:.2f}ms\n"
        f"- p95 latency: {result['p95_latency_ms']:.2f}ms\n"
        f"- Mean tokens/sec: {result['mean_tokens_per_second']:.1f}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
