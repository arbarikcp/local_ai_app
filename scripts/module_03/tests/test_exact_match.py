import pytest

from scorers.exact_match import accuracy, contains_all, exact_match, normalized_exact_match


def test_exact_match_requires_identical_strings():
    assert exact_match("Paris", "Paris") is True
    assert exact_match("Paris", "paris") is False
    assert exact_match("Paris ", "Paris") is False


def test_normalized_exact_match_tolerates_case_whitespace_and_trailing_punctuation():
    assert normalized_exact_match("  Paris.  ", "paris") is True
    assert normalized_exact_match("Paris!", "paris") is True
    assert normalized_exact_match("New   York", "new york") is True


def test_normalized_exact_match_still_rejects_different_content():
    assert normalized_exact_match("Paris", "London") is False


def test_contains_all_requires_every_substring_case_insensitive():
    text = "The invoice total is $42.50, issued to Acme Corp."
    assert contains_all(text, ["invoice", "Acme Corp", "$42.50"]) is True
    assert contains_all(text, ["invoice", "Globex"]) is False


def test_accuracy_computes_fraction_correct():
    preds = ["Paris", "London", "Berlin"]
    refs = ["paris", "london", "Madrid"]
    assert accuracy(preds, refs) == pytest.approx(2 / 3)


def test_accuracy_empty_lists_returns_zero():
    assert accuracy([], []) == 0.0


def test_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_accuracy_can_use_strict_exact_match():
    preds = ["Paris", "paris"]
    refs = ["Paris", "Paris"]
    assert accuracy(preds, refs, normalized=False) == pytest.approx(0.5)
