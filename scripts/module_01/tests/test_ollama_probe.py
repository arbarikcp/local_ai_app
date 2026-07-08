import pytest

from ollama_probe import GenerationObservation, is_ollama_available


def _make_observation(**overrides) -> GenerationObservation:
    defaults = dict(
        model="qwen2.5:1.5b",
        prompt="hello",
        response_text="hi there",
        prompt_eval_count=10,
        eval_count=20,
        total_duration_ns=3_000_000_000,
        load_duration_ns=500_000_000,
        prompt_eval_duration_ns=200_000_000,
        eval_duration_ns=2_000_000_000,
        wall_clock_seconds=3.1,
    )
    defaults.update(overrides)
    return GenerationObservation(**defaults)


def test_ttft_seconds_sums_load_and_prompt_eval_duration():
    obs = _make_observation(load_duration_ns=500_000_000, prompt_eval_duration_ns=200_000_000)
    assert obs.ttft_seconds == pytest.approx(0.7)


def test_ttft_seconds_is_none_when_metadata_missing():
    obs = _make_observation(load_duration_ns=None)
    assert obs.ttft_seconds is None


def test_tokens_per_second_computed_from_eval_count_and_duration():
    obs = _make_observation(eval_count=20, eval_duration_ns=2_000_000_000)
    assert obs.tokens_per_second == pytest.approx(10.0)


def test_tokens_per_second_is_none_when_eval_count_missing():
    obs = _make_observation(eval_count=None)
    assert obs.tokens_per_second is None


def test_tokens_per_second_is_none_when_eval_duration_zero():
    obs = _make_observation(eval_duration_ns=0)
    assert obs.tokens_per_second is None


def test_is_ollama_available_returns_false_when_unreachable():
    # No Ollama server is expected to be running against this bogus port in CI.
    assert is_ollama_available(base_url="http://localhost:1", timeout=0.2) is False
