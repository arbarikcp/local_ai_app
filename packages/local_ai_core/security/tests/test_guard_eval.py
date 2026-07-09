import pytest

from local_ai_core.security.guard_eval import LabeledExample, evaluate_guard_classifier
from local_ai_core.security.guard_pipeline import GuardDecision, GuardVerdict


class _FixedClassifier:
    """Returns a scripted verdict regardless of input text, so the
    confusion-matrix math can be tested independently of the real
    RuleBasedGuardClassifier's pattern matching.
    """

    def __init__(self, verdict: GuardVerdict) -> None:
        self.verdict = verdict

    def classify(self, text: str) -> GuardDecision:
        return GuardDecision(verdict=self.verdict, signals=[], reason="scripted")


class TestPerfectClassifier:
    def test_catches_every_malicious_example_with_no_false_positives(self):
        examples = [
            LabeledExample(text="attack 1", is_malicious=True),
            LabeledExample(text="attack 2", is_malicious=True),
            LabeledExample(text="benign 1", is_malicious=False),
        ]

        class PerfectClassifier:
            def classify(self, text: str) -> GuardDecision:
                verdict = GuardVerdict.BLOCK if "attack" in text else GuardVerdict.ALLOW
                return GuardDecision(verdict=verdict, signals=[], reason="scripted")

        report = evaluate_guard_classifier(PerfectClassifier(), examples)
        assert report.catch_rate == 1.0
        assert report.false_positive_rate == 0.0
        assert report.true_positives == 2
        assert report.true_negatives == 1


class TestClassifierThatAlwaysAllows:
    def test_catch_rate_is_zero_when_nothing_is_ever_flagged(self):
        examples = [
            LabeledExample(text="attack", is_malicious=True),
            LabeledExample(text="benign", is_malicious=False),
        ]
        report = evaluate_guard_classifier(_FixedClassifier(GuardVerdict.ALLOW), examples)
        assert report.catch_rate == 0.0
        assert report.false_positive_rate == 0.0
        assert report.false_negatives == 1


class TestClassifierThatAlwaysBlocks:
    def test_perfect_catch_rate_but_high_false_positive_rate(self):
        examples = [
            LabeledExample(text="attack", is_malicious=True),
            LabeledExample(text="benign", is_malicious=False),
        ]
        report = evaluate_guard_classifier(_FixedClassifier(GuardVerdict.BLOCK), examples)
        assert report.catch_rate == 1.0
        assert report.false_positive_rate == 1.0


class TestLatencyIsMeasured:
    def test_mean_latency_is_a_real_nonnegative_number(self):
        examples = [LabeledExample(text="x", is_malicious=False)]
        report = evaluate_guard_classifier(_FixedClassifier(GuardVerdict.ALLOW), examples)
        assert report.mean_latency_ms >= 0


class TestEmptyExamplesRaises:
    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            evaluate_guard_classifier(_FixedClassifier(GuardVerdict.ALLOW), [])


class TestVacuousRatesOnOneSidedData:
    def test_catch_rate_is_one_when_there_are_no_malicious_examples(self):
        examples = [LabeledExample(text="benign", is_malicious=False)]
        report = evaluate_guard_classifier(_FixedClassifier(GuardVerdict.ALLOW), examples)
        assert report.catch_rate == 1.0

    def test_false_positive_rate_is_zero_when_there_are_no_benign_examples(self):
        examples = [LabeledExample(text="attack", is_malicious=True)]
        report = evaluate_guard_classifier(_FixedClassifier(GuardVerdict.BLOCK), examples)
        assert report.false_positive_rate == 0.0
