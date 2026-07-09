import pytest

from local_ai_core.tracing.metrics_registry import (
    CounterSummary,
    MetricsRegistry,
    ObservableSummary,
    UnknownMetricError,
)


class TestCounters:
    def test_increment_defaults_to_one(self):
        registry = MetricsRegistry()
        registry.increment("request_count")
        registry.increment("request_count")
        summary = registry.summary()
        assert summary["request_count"] == CounterSummary(count=2)

    def test_increment_accepts_a_custom_amount(self):
        registry = MetricsRegistry()
        registry.increment("tool_call_count", amount=5)
        summary = registry.summary()
        assert summary["tool_call_count"].count == 5

    def test_incrementing_an_observable_metric_raises(self):
        registry = MetricsRegistry()
        with pytest.raises(ValueError):
            registry.increment("request_latency_ms")


class TestObservables:
    def test_observe_aggregates_mean_and_percentiles(self):
        registry = MetricsRegistry()
        for value in [10.0, 20.0, 30.0, 40.0, 50.0]:
            registry.observe("request_latency_ms", value)
        summary = registry.summary()["request_latency_ms"]
        assert isinstance(summary, ObservableSummary)
        assert summary.count == 5
        assert summary.mean == 30.0
        assert summary.p95 >= summary.p50
        assert summary.latest == 50.0

    def test_observing_a_counter_metric_raises(self):
        registry = MetricsRegistry()
        with pytest.raises(ValueError):
            registry.observe("request_count", 1.0)


class TestUnknownMetricNames:
    def test_incrementing_an_unknown_metric_raises_unknown_metric_error(self):
        registry = MetricsRegistry()
        with pytest.raises(UnknownMetricError):
            registry.increment("made_up_metric")

    def test_observing_an_unknown_metric_raises_unknown_metric_error(self):
        registry = MetricsRegistry()
        with pytest.raises(UnknownMetricError):
            registry.observe("made_up_metric", 1.0)


class TestEmptyRegistry:
    def test_summary_of_an_untouched_registry_is_empty(self):
        registry = MetricsRegistry()
        assert registry.summary() == {}


class TestAllCurriculumMetricsAreRegistered:
    def test_every_curriculum_metric_can_be_recorded_without_error(self):
        from local_ai_core.tracing.metrics_registry import METRIC_SPECS, MetricType

        registry = MetricsRegistry()
        for name, metric_type in METRIC_SPECS.items():
            if metric_type == MetricType.COUNTER:
                registry.increment(name)
            else:
                registry.observe(name, 1.0)

        assert len(registry.summary()) == len(METRIC_SPECS)
