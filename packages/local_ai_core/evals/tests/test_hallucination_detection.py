import pytest

from local_ai_core.evals.hallucination_detection import compute_auroc


class TestComputeAuroc:
    def test_perfect_separation_scores_one(self):
        assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_perfectly_reversed_scores_zero(self):
        assert compute_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_all_scores_tied_is_chance_level(self):
        assert compute_auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_no_positive_examples_returns_chance_level(self):
        assert compute_auroc([0, 0], [0.1, 0.9]) == 0.5

    def test_no_negative_examples_returns_chance_level(self):
        assert compute_auroc([1, 1], [0.1, 0.9]) == 0.5

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_auroc([0, 1], [0.5])

    def test_partial_separation_scores_between_chance_and_perfect(self):
        # One negative outscores one positive, otherwise correctly ordered.
        auroc = compute_auroc([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9])
        assert 0.5 < auroc < 1.0
