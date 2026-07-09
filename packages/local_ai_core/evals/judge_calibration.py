"""Judge-human agreement (theory doc, "The judge-model problem" required
lesson): measure judge-human agreement before trusting an LLM judge.
Simple agreement for early labs, Cohen's kappa for more formal evaluation
- an unvalidated judge is a random number generator with fluent
explanations. Both are pure, deterministic statistics; no model needed to
compute either.
"""

from __future__ import annotations


def simple_agreement(judge_labels: list[bool], human_labels: list[bool]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    if not judge_labels:
        return 0.0
    matches = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    return matches / len(judge_labels)


def cohens_kappa(judge_labels: list[bool], human_labels: list[bool]) -> float:
    """kappa = (p_observed - p_expected) / (1 - p_expected) - corrects
    simple agreement for the agreement expected by chance alone, given
    each rater's own label distribution. Two raters that both label
    everything True agree 100% of the time by simple agreement, but that
    agreement is meaningless; kappa discounts it.

    Returns 1.0 in the degenerate case where p_expected == 1.0 (both
    raters have zero variance and agree completely) - there is no
    disagreement left to explain by chance, so perfect agreement is the
    only well-defined value, not an error.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_observed = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n

    judge_true_rate = sum(judge_labels) / n
    human_true_rate = sum(human_labels) / n
    p_expected = judge_true_rate * human_true_rate + (1 - judge_true_rate) * (1 - human_true_rate)

    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)
