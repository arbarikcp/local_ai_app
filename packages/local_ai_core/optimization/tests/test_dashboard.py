from local_ai_core.optimization.dashboard import InMemoryMetricsHook, PerformanceDashboard
from local_ai_core.runtimes.errors import RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

REQUEST = LLMRequest(model="test-model", prompt="hello there friend")


class TestEmptyDashboard:
    def test_summary_of_no_requests_is_all_zeros(self):
        dashboard = PerformanceDashboard()
        summary = dashboard.summary()
        assert summary.request_count == 0
        assert summary.error_rate == 0.0


class TestDashboardAggregatesRealRequests:
    async def test_successful_requests_are_recorded_with_no_errors(self):
        hook = InMemoryMetricsHook()
        runtime = FakeRuntime(default_response="one two three", simulated_latency_ms=5, metrics_hook=hook)
        dashboard = PerformanceDashboard(hook=hook)

        for _ in range(3):
            await runtime.generate(REQUEST)

        summary = dashboard.summary()
        assert summary.request_count == 3
        assert summary.error_count == 0
        assert summary.error_rate == 0.0
        assert summary.mean_latency_ms > 0
        assert summary.mean_tokens_per_second > 0

    async def test_failures_are_counted_in_the_error_rate(self):
        hook = InMemoryMetricsHook()
        runtime = FakeRuntime(fail_with=RuntimeUnavailable("down"), metrics_hook=hook)
        dashboard = PerformanceDashboard(hook=hook)

        for _ in range(2):
            try:
                await runtime.generate(REQUEST)
            except RuntimeUnavailable:
                pass

        summary = dashboard.summary()
        assert summary.request_count == 2
        assert summary.error_count == 2
        assert summary.error_rate == 1.0

    async def test_mixed_success_and_failure_produces_a_partial_error_rate(self):
        hook = InMemoryMetricsHook()
        healthy = FakeRuntime(default_response="ok", metrics_hook=hook)
        broken = FakeRuntime(fail_with=RuntimeUnavailable("down"), metrics_hook=hook)
        dashboard = PerformanceDashboard(hook=hook)

        await healthy.generate(REQUEST)
        try:
            await broken.generate(REQUEST)
        except RuntimeUnavailable:
            pass

        summary = dashboard.summary()
        assert summary.request_count == 2
        assert summary.error_count == 1
        assert summary.error_rate == 0.5


class TestPercentiles:
    async def test_p95_is_never_less_than_p50(self):
        hook = InMemoryMetricsHook()
        runtime = FakeRuntime(default_response="ok", simulated_latency_ms=1, metrics_hook=hook)
        dashboard = PerformanceDashboard(hook=hook)

        for _ in range(10):
            await runtime.generate(REQUEST)

        summary = dashboard.summary()
        assert summary.p95_latency_ms >= summary.p50_latency_ms
