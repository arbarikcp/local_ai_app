import pytest

import lab_measure_concurrency as sut
from local_ai_core.gateway.admission_control import ConcurrencyMeasurement
from local_ai_core.runtimes.errors import RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime


class TestPercentile:
    def test_p95_of_sorted_list(self):
        values = list(range(1, 101))  # 1..100
        assert sut.percentile(values, 0.95) == pytest.approx(96, abs=1)

    def test_empty_list_returns_zero(self):
        assert sut.percentile([], 0.95) == 0.0

    def test_single_value_returns_that_value(self):
        assert sut.percentile([42.0], 0.95) == 42.0

    def test_median_of_small_list(self):
        assert sut.percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)


class TestMeasureConcurrencyLevel:
    async def test_measures_latency_across_all_requests(self):
        runtime = FakeRuntime(simulated_latency_ms=1.0)
        measurement = await sut.measure_concurrency_level(runtime, "m", "p", n_concurrent=2, n_requests=6)
        assert measurement.concurrency == 2
        assert measurement.mean_latency_seconds > 0
        assert measurement.failure_rate == 0.0

    async def test_records_failures_without_aborting_the_batch(self):
        runtime = FakeRuntime(fail_first_n_calls=3, transient_error_for_first_n=RuntimeUnavailable("down"))
        measurement = await sut.measure_concurrency_level(runtime, "m", "p", n_concurrent=1, n_requests=6)
        assert measurement.failure_rate == pytest.approx(3 / 6)

    async def test_all_requests_get_admitted_when_queue_is_large_enough(self):
        runtime = FakeRuntime(simulated_latency_ms=0.0)
        measurement = await sut.measure_concurrency_level(runtime, "m", "p", n_concurrent=1, n_requests=10)
        assert measurement.failure_rate == 0.0  # queue sized to n_requests, nothing should be rejected


class TestMeasurementsToMarkdownTable:
    def test_renders_all_levels(self):
        measurements = [
            ConcurrencyMeasurement(concurrency=1, mean_latency_seconds=0.5, p95_latency_seconds=0.6, failure_rate=0.0),
            ConcurrencyMeasurement(concurrency=4, mean_latency_seconds=1.2, p95_latency_seconds=2.0, failure_rate=0.1),
        ]
        table = sut.measurements_to_markdown_table(measurements)
        assert "| 1 |" in table
        assert "| 4 |" in table
        assert "10%" in table


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--model", "qwen2.5:3b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
