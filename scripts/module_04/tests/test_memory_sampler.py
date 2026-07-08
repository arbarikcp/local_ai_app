import os
import subprocess
import time

from memory_sampler import PeakMemorySampler, find_pid_by_name, get_rss_bytes, sample_peak_rss_during


def test_find_pid_by_name_returns_none_for_a_process_that_does_not_exist():
    assert find_pid_by_name("definitely-not-a-real-process-xyz") is None


def test_find_pid_by_name_finds_a_real_running_process():
    # `sleep` is a stable, always-present binary to search for by exact name.
    proc = subprocess.Popen(["sleep", "2"])
    try:
        time.sleep(0.2)  # let it register
        found_pid = find_pid_by_name("sleep")
        assert found_pid is not None
    finally:
        proc.terminate()
        proc.wait()


def test_get_rss_bytes_returns_a_plausible_positive_value_for_current_process():
    rss = get_rss_bytes(os.getpid())
    assert rss is not None
    # A running Python test process should be at least a few hundred KB and
    # well under, say, 10 GB - a sanity range, not a precise assertion.
    assert 100_000 < rss < 10 * 1024**3


def test_get_rss_bytes_returns_none_for_a_process_that_has_already_exited():
    proc = subprocess.Popen(["true"])
    proc.wait()
    # Give the OS a moment to reap it; `ps` should no longer find this pid.
    time.sleep(0.2)
    assert get_rss_bytes(proc.pid) is None


def test_peak_memory_sampler_tracks_a_peak_for_the_current_process():
    sampler = PeakMemorySampler(os.getpid(), poll_interval_seconds=0.02)
    sampler.start()
    time.sleep(0.1)
    peak = sampler.stop()
    assert peak is not None
    assert peak > 0


def test_peak_memory_sampler_as_context_manager():
    with PeakMemorySampler(os.getpid(), poll_interval_seconds=0.02) as sampler:
        time.sleep(0.1)
    assert sampler.peak_bytes is not None
    assert sampler.peak_bytes > 0


def test_peak_memory_sampler_peak_is_none_before_start():
    sampler = PeakMemorySampler(os.getpid())
    assert sampler.peak_bytes is None


def test_sample_peak_rss_during_returns_fn_result_and_a_peak():
    def fn():
        time.sleep(0.1)
        return "done"

    result, peak = sample_peak_rss_during(os.getpid(), fn, poll_interval_seconds=0.02)
    assert result == "done"
    assert peak is not None
    assert peak > 0


def test_sample_peak_rss_during_propagates_exceptions_from_fn():
    def fn():
        raise ValueError("boom")

    try:
        sample_peak_rss_during(os.getpid(), fn, poll_interval_seconds=0.02)
        assert False, "expected ValueError to propagate"
    except ValueError as exc:
        assert "boom" in str(exc)
