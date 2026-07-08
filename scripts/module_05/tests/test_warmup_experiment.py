import pytest

from warmup_experiment import WarmupResult, result_to_markdown, run_warmup_experiment


def test_run_warmup_experiment_calls_fn_once_cold_and_n_times_warm():
    calls = []

    def fake_get_ttft(model, prompt):
        calls.append((model, prompt))
        return 1.0 if len(calls) == 1 else 0.1

    result = run_warmup_experiment("m", "p", fake_get_ttft, n_warm_calls=3)
    assert len(calls) == 4  # 1 cold + 3 warm
    assert result.cold_ttft_seconds == pytest.approx(1.0)
    assert result.warm_ttft_seconds == [pytest.approx(0.1)] * 3


def test_run_warmup_experiment_rejects_invalid_n_warm_calls():
    with pytest.raises(ValueError):
        run_warmup_experiment("m", "p", lambda m, p: 1.0, n_warm_calls=0)


class TestWarmupResultProperties:
    def test_mean_warm_ttft_averages_successful_calls(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[0.1, 0.2, 0.3])
        assert result.mean_warm_ttft_seconds == pytest.approx(0.2)

    def test_mean_warm_ttft_ignores_failed_calls(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[0.1, None, 0.3])
        assert result.mean_warm_ttft_seconds == pytest.approx(0.2)

    def test_mean_warm_ttft_none_when_all_failed(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[None, None])
        assert result.mean_warm_ttft_seconds is None

    def test_speedup_factor_computed_correctly(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[0.25, 0.25])
        assert result.speedup_factor == pytest.approx(4.0)

    def test_speedup_factor_none_when_cold_missing(self):
        result = WarmupResult(model="m", cold_ttft_seconds=None, warm_ttft_seconds=[0.1])
        assert result.speedup_factor is None

    def test_speedup_factor_none_when_no_successful_warm_calls(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[None])
        assert result.speedup_factor is None

    def test_successful_warm_ttfts_filters_none(self):
        result = WarmupResult(model="m", cold_ttft_seconds=1.0, warm_ttft_seconds=[0.1, None, 0.2])
        assert result.successful_warm_ttfts == [0.1, 0.2]


class TestResultToMarkdown:
    def test_renders_all_fields_on_success(self):
        result = WarmupResult(model="qwen2.5:3b", cold_ttft_seconds=1.0, warm_ttft_seconds=[0.2, 0.2])
        md = result_to_markdown(result)
        assert "qwen2.5:3b" in md
        assert "1.000s" in md
        assert "5.00x" in md

    def test_renders_failed_for_none_values(self):
        result = WarmupResult(model="m", cold_ttft_seconds=None, warm_ttft_seconds=[None])
        md = result_to_markdown(result)
        assert "failed" in md
        assert "n/a" in md
