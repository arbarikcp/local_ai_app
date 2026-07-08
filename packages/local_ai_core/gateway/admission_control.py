"""AdmissionPolicy / AdmissionController — turns "max_concurrent_requests: 1
is often correct" from folklore into an explicit, on-record decision
(theory doc "Why max_concurrent_requests: 1 is often correct").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from .queue import BoundedRequestQueue, QueuedResult

T = TypeVar("T")

DEFAULT_MAX_QUEUE_SIZE = 10
_DEFAULT_REASON = "default: single unified-memory Mac, not yet measured (see theory doc)"


@dataclass(frozen=True)
class AdmissionPolicy:
    """An explicit, documented concurrency decision - never a bare number.

    The default (max_concurrent_requests=1) is this course's honest
    starting point: safe on a single unified-memory Mac, unmeasured. Any
    higher value must cite the measurement that justified it in ``reason``
    - the constructor enforces this rather than trusting callers to
    remember, so the policy is always a decision on record.
    """

    max_concurrent_requests: int = 1
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    reason: str = field(default=_DEFAULT_REASON)

    def __post_init__(self) -> None:
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        if self.max_concurrent_requests > 1 and self.reason == _DEFAULT_REASON:
            raise ValueError(
                "max_concurrent_requests > 1 requires a specific `reason` citing the "
                "measurement that justified it - the default reason is not enough. "
                "See recommend_policy_from_measurements() for how to derive one."
            )


class AdmissionController:
    """Wraps a BoundedRequestQueue with an explicit, named AdmissionPolicy."""

    def __init__(self, policy: AdmissionPolicy | None = None) -> None:
        self.policy = policy or AdmissionPolicy()
        self._queue = BoundedRequestQueue(
            max_concurrent=self.policy.max_concurrent_requests,
            max_queue_size=self.policy.max_queue_size,
        )

    async def submit(self, fn: Callable[[], Awaitable[T]]) -> QueuedResult[T]:
        return await self._queue.submit(fn)

    @property
    def waiting_count(self) -> int:
        return self._queue.waiting_count

    @property
    def running_count(self) -> int:
        return self._queue.running_count


@dataclass(frozen=True)
class ConcurrencyMeasurement:
    """One row of Lab 1-3's output: results of running at one concurrency level."""

    concurrency: int
    mean_latency_seconds: float
    p95_latency_seconds: float
    failure_rate: float


def recommend_policy_from_measurements(
    measurements: list[ConcurrencyMeasurement],
    *,
    max_p95_growth_factor: float = 2.0,
    max_acceptable_failure_rate: float = 0.0,
) -> AdmissionPolicy:
    """Pick the highest concurrency level whose p95 latency doesn't exceed
    ``max_p95_growth_factor`` times the concurrency=1 baseline, and whose
    failure rate stays at or below ``max_acceptable_failure_rate``.

    Encodes the theory doc's rule precisely: "increase to 2 only after
    measurement," where measurement means p95 (not just mean) latency
    didn't blow up and nothing started failing. Assumes measurements are
    roughly monotonic in concurrency (latency/failure rate non-decreasing
    as concurrency rises) and stops at the first level that fails either
    check, rather than cherry-picking a later level that happens to look
    good again.
    """
    if not measurements:
        return AdmissionPolicy()  # no data -> the safe default

    by_concurrency = {m.concurrency: m for m in measurements}
    baseline = by_concurrency.get(1)
    if baseline is None:
        return AdmissionPolicy()  # can't compare against a baseline we don't have

    best = 1
    for concurrency in sorted(c for c in by_concurrency if c != 1):
        m = by_concurrency[concurrency]
        p95_ok = m.p95_latency_seconds <= baseline.p95_latency_seconds * max_p95_growth_factor
        failure_ok = m.failure_rate <= max_acceptable_failure_rate
        if p95_ok and failure_ok:
            best = concurrency
        else:
            break

    if best == 1:
        return AdmissionPolicy()

    chosen = by_concurrency[best]
    reason = (
        f"measured: concurrency={best} kept p95 latency at {chosen.p95_latency_seconds:.2f}s "
        f"(baseline {baseline.p95_latency_seconds:.2f}s at concurrency=1, within "
        f"{max_p95_growth_factor}x) with {chosen.failure_rate:.0%} failure rate"
    )
    return AdmissionPolicy(max_concurrent_requests=best, reason=reason)
