"""Hallucination detection as binary classification (theory doc §9, Lab 8)
- AUROC measures how well a detector score (e.g.
`citation_faithfulness_score`) separates known-grounded from
known-hallucinated answers, across every possible threshold at once,
rather than picking one threshold and reporting an accuracy that depends
on the choice.

Implemented from scratch (rank-based Mann-Whitney U formulation) rather
than adding `scikit-learn` as a dependency for one function - the math is
a few lines: AUROC equals the probability that a randomly chosen positive
example scores higher than a randomly chosen negative example, which the
rank-sum formulation computes exactly without iterating over thresholds.
"""

from __future__ import annotations


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """`labels`: 1 for a known-hallucinated (positive) case, 0 for a
    known-grounded (negative) case. `scores`: the detector's score for
    that same case (higher = more hallucinated, matching the label
    convention). Returns 0.5 (chance) if there are no positive or no
    negative examples - AUROC is undefined without both classes, and 0.5
    is the honest "no information" value rather than raising.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")

    positives = [s for lbl, s in zip(labels, scores) if lbl == 1]
    negatives = [s for lbl, s in zip(labels, scores) if lbl == 0]
    if not positives or not negatives:
        return 0.5

    # Rank-sum (Mann-Whitney U) formulation: average ranks for ties, then
    # AUROC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    all_scores = sorted(scores)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(all_scores):
        j = i
        while j < len(all_scores) and all_scores[j] == all_scores[i]:
            j += 1
        average_rank = (i + 1 + j) / 2.0  # 1-indexed rank average over the tied block
        for value in all_scores[i:j]:
            ranks[value] = average_rank
        i = j

    positive_rank_sum = sum(ranks[s] for s in positives)
    n_pos, n_neg = len(positives), len(negatives)
    u_statistic = positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)
