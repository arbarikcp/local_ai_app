import pytest

from token_estimate import (
    HFTokenizerCounter,
    TokenizerUnavailable,
    estimate_tokens_heuristic,
    words_for_target_tokens,
)


def test_estimate_tokens_heuristic_scales_with_word_count():
    short = estimate_tokens_heuristic("one two three")
    long = estimate_tokens_heuristic(" ".join(["word"] * 100))
    assert short < long
    assert long == pytest.approx(130, abs=5)


def test_estimate_tokens_heuristic_minimum_is_one():
    assert estimate_tokens_heuristic("") == 1
    assert estimate_tokens_heuristic("hi") >= 1


def test_words_for_target_tokens_round_trips_approximately():
    target = 500
    words = words_for_target_tokens(target)
    text = " ".join(["word"] * words)
    estimated = estimate_tokens_heuristic(text)
    assert estimated == pytest.approx(target, rel=0.05)


def test_words_for_target_tokens_rejects_non_positive():
    with pytest.raises(ValueError):
        words_for_target_tokens(0)
    with pytest.raises(ValueError):
        words_for_target_tokens(-10)


def test_hf_tokenizer_counter_raises_clear_error_when_unavailable():
    counter = HFTokenizerCounter(model_id="definitely-not-a-real-model-id/xyz")
    with pytest.raises(TokenizerUnavailable):
        counter.count("hello world")


def test_tokenizer_unavailable_message_mentions_model_id():
    err = TokenizerUnavailable(model_id="some/model")
    assert "some/model" in str(err)
