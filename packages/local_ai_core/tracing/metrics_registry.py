"""MetricsRegistry — real, in-memory implementation of curriculum's exact
metric table (theory doc §2, §6-7). Metric names are constrained to the
known set below - a typo raises loudly at the call site instead of quietly
creating a new, disconnected metric no dashboard will ever look at, the
same discipline Module 20's `route_model()` gave routing reasons: every
decision is traceable, not ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    OBSERVABLE = "observable"  # covers curriculum's "histogram" and "gauge" kinds


METRIC_SPECS: dict[str, MetricType] = {
    "request_count": MetricType.COUNTER,
    "request_latency_ms": MetricType.OBSERVABLE,
    "ttft_ms": MetricType.OBSERVABLE,
    "tokens_per_second": MetricType.OBSERVABLE,
    "prompt_tokens": MetricType.OBSERVABLE,
    "completion_tokens": MetricType.OBSERVABLE,
    "invalid_json_count": MetricType.COUNTER,
    "retrieval_recall_estimate": MetricType.OBSERVABLE,
    "tool_call_count": MetricType.COUNTER,
    "tool_error_count": MetricType.COUNTER,
    "policy_violation_count": MetricType.COUNTER,
    "fallback_count": MetricType.COUNTER,
}


class UnknownMetricError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown metric {name!r} - not in curriculum's metric table")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile * (len(ordered) - 1)))
    return ordered[index]


@dataclass(frozen=True)
class CounterSummary:
    count: int


@dataclass(frozen=True)
class ObservableSummary:
    count: int
    mean: float
    p50: float
    p95: float
    latest: float


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._observations: dict[str, list[float]] = {}

    def _require_type(self, name: str, expected: MetricType) -> None:
        spec = METRIC_SPECS.get(name)
        if spec is None:
            raise UnknownMetricError(name)
        if spec != expected:
            raise ValueError(f"Metric {name!r} is a {spec.value}, not a {expected.value}")

    def increment(self, name: str, amount: int = 1) -> None:
        self._require_type(name, MetricType.COUNTER)
        self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        self._require_type(name, MetricType.OBSERVABLE)
        self._observations.setdefault(name, []).append(value)

    def summary(self) -> dict[str, CounterSummary | ObservableSummary]:
        result: dict[str, CounterSummary | ObservableSummary] = {}
        for name, count in self._counters.items():
            result[name] = CounterSummary(count=count)
        for name, values in self._observations.items():
            if not values:
                continue
            result[name] = ObservableSummary(
                count=len(values),
                mean=sum(values) / len(values),
                p50=_percentile(values, 0.50),
                p95=_percentile(values, 0.95),
                latest=values[-1],
            )
        return result
