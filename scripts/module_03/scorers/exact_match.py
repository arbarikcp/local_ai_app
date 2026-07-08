"""Exact and normalized-match scorers.

Precise but brittle (docs/modules/03_local_model_selection_and_benchmarking.md
§9): use these for tasks with a single unambiguous correct string
(classification labels, short extraction fields), not for open-ended
generation.
"""

from __future__ import annotations

import re


def exact_match(prediction: str, reference: str) -> bool:
    """Byte-for-byte match. The strictest, most brittle scorer."""
    return prediction == reference


def normalized_exact_match(prediction: str, reference: str) -> bool:
    """Match after lowercasing, trimming, and collapsing internal whitespace.

    Tolerates the most common harmless variation (trailing period, extra
    spaces, casing) without tolerating actually-different content.
    """
    return _normalize(prediction) == _normalize(reference)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".!?")
    return text


def contains_all(prediction: str, required_substrings: list[str]) -> bool:
    """True if every required substring appears (case-insensitive) in prediction.

    Useful for tasks where several facts must appear but exact phrasing is
    expected to vary (e.g. summarization must mention certain entities).
    """
    lowered = prediction.lower()
    return all(s.lower() in lowered for s in required_substrings)


def accuracy(predictions: list[str], references: list[str], *, normalized: bool = True) -> float:
    """Fraction of predictions matching their reference, using the chosen match rule."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return 0.0
    match_fn = normalized_exact_match if normalized else exact_match
    correct = sum(1 for p, r in zip(predictions, references) if match_fn(p, r))
    return correct / len(predictions)
