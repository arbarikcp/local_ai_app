"""Real catch-rate/false-positive-rate/latency measurement for a
`GuardClassifier` against a labeled red-team set (theory doc Labs 6-7).
Same evaluation discipline as Module 13's `answer_metrics` and Module 19's
`evaluation.py`: real aggregate numbers over real classifier calls, not
hand-asserted percentages.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.runtimes.base import Timer
from local_ai_core.security.guard_pipeline import GuardClassifier, GuardVerdict


@dataclass(frozen=True)
class LabeledExample:
    text: str
    is_malicious: bool
    label: str = ""


@dataclass(frozen=True)
class GuardEvalReport:
    total: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    mean_latency_ms: float

    @property
    def catch_rate(self) -> float:
        """Recall: of the genuinely malicious examples, what fraction did
        the classifier flag (verdict != ALLOW)? 1.0 vacuously when there
        were no malicious examples to catch.
        """
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def false_positive_rate(self) -> float:
        """Of the genuinely benign examples, what fraction did the
        classifier wrongly flag? 0.0 vacuously when there were no benign
        examples.
        """
        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator else 0.0


def evaluate_guard_classifier(classifier: GuardClassifier, examples: list[LabeledExample]) -> GuardEvalReport:
    if not examples:
        raise ValueError("examples must not be empty")

    true_positives = false_positives = true_negatives = false_negatives = 0
    latencies: list[float] = []

    for example in examples:
        timer = Timer()
        decision = classifier.classify(example.text)
        latencies.append(timer.elapsed_ms)

        predicted_malicious = decision.verdict != GuardVerdict.ALLOW
        if example.is_malicious and predicted_malicious:
            true_positives += 1
        elif example.is_malicious and not predicted_malicious:
            false_negatives += 1
        elif not example.is_malicious and predicted_malicious:
            false_positives += 1
        else:
            true_negatives += 1

    return GuardEvalReport(
        total=len(examples),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        mean_latency_ms=sum(latencies) / len(latencies),
    )
