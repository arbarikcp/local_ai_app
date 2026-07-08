import pytest

from local_ai_core.gateway.admission_control import (
    AdmissionController,
    AdmissionPolicy,
    ConcurrencyMeasurement,
    recommend_policy_from_measurements,
)
from local_ai_core.gateway.queue import QueueFullError


class TestAdmissionPolicy:
    def test_default_is_concurrency_one_with_the_default_reason(self):
        policy = AdmissionPolicy()
        assert policy.max_concurrent_requests == 1
        assert "not yet measured" in policy.reason

    def test_rejects_concurrency_below_one(self):
        with pytest.raises(ValueError):
            AdmissionPolicy(max_concurrent_requests=0)

    def test_rejects_negative_queue_size(self):
        with pytest.raises(ValueError):
            AdmissionPolicy(max_queue_size=-1)

    def test_concurrency_above_one_requires_a_specific_reason(self):
        with pytest.raises(ValueError, match="requires a specific"):
            AdmissionPolicy(max_concurrent_requests=2)  # default reason, no override

    def test_concurrency_above_one_with_explicit_reason_succeeds(self):
        policy = AdmissionPolicy(max_concurrent_requests=2, reason="measured: p95 stayed flat at c=2")
        assert policy.max_concurrent_requests == 2

    def test_concurrency_of_one_never_requires_a_special_reason(self):
        policy = AdmissionPolicy(max_concurrent_requests=1)  # default reason is fine here
        assert policy.max_concurrent_requests == 1


class TestAdmissionController:
    async def test_delegates_to_an_internal_bounded_queue(self):
        controller = AdmissionController()

        async def fn():
            return "ok"

        result = await controller.submit(fn)
        assert result.result == "ok"

    async def test_uses_the_policy_concurrency_limit(self):
        policy = AdmissionPolicy(max_concurrent_requests=2, max_queue_size=0, reason="measured: test setup")
        controller = AdmissionController(policy)
        assert controller._queue.max_concurrent == 2
        assert controller._queue.max_queue_size == 0

    async def test_rejects_beyond_capacity_via_queue_full_error(self):
        import asyncio

        policy = AdmissionPolicy(max_concurrent_requests=1, max_queue_size=0)
        controller = AdmissionController(policy)
        release = asyncio.Event()

        async def blocking_fn():
            await release.wait()
            return "ok"

        first_task = asyncio.create_task(controller.submit(blocking_fn))
        await asyncio.sleep(0.01)

        async def instant_fn():
            return "rejected"

        with pytest.raises(QueueFullError):
            await controller.submit(instant_fn)

        release.set()
        await first_task

    async def test_running_and_waiting_counts_reflect_the_underlying_queue(self):
        controller = AdmissionController()
        assert controller.running_count == 0
        assert controller.waiting_count == 0


class TestRecommendPolicyFromMeasurements:
    def test_empty_measurements_returns_the_safe_default(self):
        policy = recommend_policy_from_measurements([])
        assert policy.max_concurrent_requests == 1

    def test_missing_baseline_returns_the_safe_default(self):
        measurements = [ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=1, p95_latency_seconds=1, failure_rate=0)]
        policy = recommend_policy_from_measurements(measurements)
        assert policy.max_concurrent_requests == 1

    def test_recommends_higher_concurrency_when_p95_stays_flat(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=1.0, p95_latency_seconds=1.5, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=1.1, p95_latency_seconds=1.6, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=4, mean_latency_seconds=1.2, p95_latency_seconds=1.8, failure_rate=0.0),
        ]
        policy = recommend_policy_from_measurements(measurements)
        assert policy.max_concurrent_requests == 4
        assert "measured" in policy.reason
        assert "concurrency=4" in policy.reason

    def test_stays_at_one_when_p95_blows_up_immediately(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=1.0, p95_latency_seconds=1.5, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=3.0, p95_latency_seconds=10.0, failure_rate=0.0),
        ]
        policy = recommend_policy_from_measurements(measurements)
        assert policy.max_concurrent_requests == 1

    def test_stops_at_first_level_that_fails_even_if_a_later_one_looks_fine(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=1.0, p95_latency_seconds=1.0, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=5.0, p95_latency_seconds=10.0, failure_rate=0.0),  # fails
            ConcurrencyMeasurement(concurrency=4, mean_latency_seconds=1.0, p95_latency_seconds=1.1, failure_rate=0.0),  # would pass alone
        ]
        policy = recommend_policy_from_measurements(measurements)
        assert policy.max_concurrent_requests == 1  # stopped at concurrency=2's failure, never considered 4

    def test_nonzero_failure_rate_blocks_recommendation_by_default(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=1.0, p95_latency_seconds=1.0, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=1.0, p95_latency_seconds=1.0, failure_rate=0.05),
        ]
        policy = recommend_policy_from_measurements(measurements)
        assert policy.max_concurrent_requests == 1

    def test_custom_growth_factor_and_failure_tolerance_are_honored(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=1.0, p95_latency_seconds=1.0, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=2, mean_latency_seconds=1.0, p95_latency_seconds=2.5, failure_rate=0.1),
        ]
        strict = recommend_policy_from_measurements(measurements)
        assert strict.max_concurrent_requests == 1

        lenient = recommend_policy_from_measurements(
            measurements, max_p95_growth_factor=3.0, max_acceptable_failure_rate=0.2
        )
        assert lenient.max_concurrent_requests == 2
