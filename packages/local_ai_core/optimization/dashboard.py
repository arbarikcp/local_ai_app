"""Performance dashboard (theory doc Lab 7) — `InMemoryMetricsHook`
implements the existing `MetricsHook` Protocol (Module 6's `base.py`) to
collect real per-request records, and `PerformanceDashboard` aggregates
them into the same p50/p95/error-rate summary shape Module 6.5's
`ConcurrencyMeasurement` already established for benchmark rows - applied
here to live request traffic instead of a one-off benchmark run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from local_ai_core.runtimes.errors import LLMError
from local_ai_core.runtimes.types import LLMRequest, LLMResponse


@dataclass(frozen=True)
class RequestRecord:
    model: str
    latency_ms: float
    completion_tokens: int | None
    succeeded: bool
    error_type: str | None


class InMemoryMetricsHook:
    """A real `MetricsHook` implementation (structurally satisfies Module
    6's Protocol) that keeps every request's outcome in memory for later
    aggregation - the same role `LoggingMetricsHook` plays for logs, but for
    a dashboard.
    """

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []

    def on_request(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        error: LLMError | None,
        latency_ms: float,
    ) -> None:
        self.records.append(
            RequestRecord(
                model=request.model,
                latency_ms=latency_ms,
                completion_tokens=response.completion_tokens if response else None,
                succeeded=error is None,
                error_type=type(error).__name__ if error else None,
            )
        )


@dataclass(frozen=True)
class DashboardSummary:
    request_count: int
    error_count: int
    error_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile * (len(ordered) - 1)))
    return ordered[index]


@dataclass
class PerformanceDashboard:
    hook: InMemoryMetricsHook = field(default_factory=InMemoryMetricsHook)

    def summary(self) -> DashboardSummary:
        records = self.hook.records
        if not records:
            return DashboardSummary(
                request_count=0,
                error_count=0,
                error_rate=0.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                mean_tokens_per_second=0.0,
            )

        latencies = [r.latency_ms for r in records]
        error_count = sum(1 for r in records if not r.succeeded)

        tokens_per_second_samples = [
            r.completion_tokens / (r.latency_ms / 1000)
            for r in records
            if r.succeeded and r.completion_tokens and r.latency_ms > 0
        ]

        return DashboardSummary(
            request_count=len(records),
            error_count=error_count,
            error_rate=error_count / len(records),
            mean_latency_ms=sum(latencies) / len(latencies),
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            mean_tokens_per_second=(
                sum(tokens_per_second_samples) / len(tokens_per_second_samples)
                if tokens_per_second_samples
                else 0.0
            ),
        )
