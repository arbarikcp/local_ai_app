import pytest

import lab_4_3_concurrency_simulation as sut


def test_concurrency_level_result_mean_and_max_latency():
    result = sut.ConcurrencyLevelResult(
        n_concurrent=3, batch_wall_clock_seconds=2.0, request_latencies_seconds=[1.0, 2.0, 3.0]
    )
    assert result.mean_latency_seconds == pytest.approx(2.0)
    assert result.max_latency_seconds == pytest.approx(3.0)


def test_concurrency_level_result_latency_properties_none_when_empty():
    result = sut.ConcurrencyLevelResult(n_concurrent=2, batch_wall_clock_seconds=1.0)
    assert result.mean_latency_seconds is None
    assert result.max_latency_seconds is None


def test_failure_rate_computed_against_n_concurrent():
    result = sut.ConcurrencyLevelResult(
        n_concurrent=4, batch_wall_clock_seconds=1.0, request_latencies_seconds=[1.0, 1.0], failure_count=2
    )
    assert result.failure_rate == pytest.approx(0.5)


def test_failure_rate_zero_when_no_failures():
    result = sut.ConcurrencyLevelResult(n_concurrent=4, batch_wall_clock_seconds=1.0)
    assert result.failure_rate == 0.0


def test_run_concurrency_level_collects_all_successful_latencies(monkeypatch):
    def fake_generate(model, prompt, timeout=60.0):
        return object()  # _fire_one_request only needs the call to not raise

    monkeypatch.setattr(sut, "generate", fake_generate)
    monkeypatch.setattr(sut, "find_pid_by_name", lambda name: None)

    result = sut.run_concurrency_level("fake-model", "prompt", n_concurrent=4)
    assert len(result.request_latencies_seconds) == 4
    assert result.failure_count == 0
    assert result.peak_rss_bytes is None


def test_run_concurrency_level_counts_failures_without_aborting(monkeypatch):
    call_count = {"n": 0}

    def flaky_generate(model, prompt, timeout=60.0):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            raise RuntimeError("simulated failure")
        return object()

    monkeypatch.setattr(sut, "generate", flaky_generate)
    monkeypatch.setattr(sut, "find_pid_by_name", lambda name: None)

    result = sut.run_concurrency_level("fake-model", "prompt", n_concurrent=4)
    assert result.failure_count == 2
    assert len(result.request_latencies_seconds) == 2


def test_results_to_markdown_table_renders_all_levels():
    results = [
        sut.ConcurrencyLevelResult(n_concurrent=1, batch_wall_clock_seconds=1.0, request_latencies_seconds=[1.0]),
        sut.ConcurrencyLevelResult(
            n_concurrent=8, batch_wall_clock_seconds=5.0, request_latencies_seconds=[4.0], failure_count=7,
            peak_rss_bytes=4 * 1024**3,
        ),
    ]
    table = sut.results_to_markdown_table(results)
    assert "| 1 |" in table
    assert "| 8 |" in table
    assert "88%" in table  # 7/8 failure rate
    assert "4.00 GiB" in table
