import pytest

from local_ai_core.evals.judge_calibration import cohens_kappa, simple_agreement


class TestSimpleAgreement:
    def test_perfect_agreement_scores_one(self):
        assert simple_agreement([True, False, True], [True, False, True]) == 1.0

    def test_no_agreement_scores_zero(self):
        assert simple_agreement([True, True], [False, False]) == 0.0

    def test_partial_agreement(self):
        assert simple_agreement([True, True, False], [True, False, False]) == pytest.approx(2 / 3)

    def test_empty_input_returns_zero(self):
        assert simple_agreement([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            simple_agreement([True], [True, False])


class TestCohensKappa:
    def test_perfect_agreement_scores_one(self):
        labels = [True, True, False, False]
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_known_worked_example(self):
        # p_o = 0.8 (8/10 match), p_e = 0.5 (both raters 50% true rate) -> kappa = 0.6
        human = [True, True, True, True, True, False, False, False, False, False]
        judge = [True, True, True, True, False, False, False, False, False, True]
        assert cohens_kappa(judge, human) == pytest.approx(0.6)

    def test_kappa_is_lower_than_simple_agreement_when_chance_agreement_is_high(self):
        human = [True, True, True, True, True, False, False, False, False, False]
        judge = [True, True, True, True, False, False, False, False, False, True]
        assert cohens_kappa(judge, human) < simple_agreement(judge, human)

    def test_both_raters_always_true_is_the_degenerate_perfect_case(self):
        assert cohens_kappa([True, True], [True, True]) == 1.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([True], [True, False])

    def test_empty_input_returns_zero(self):
        assert cohens_kappa([], []) == 0.0
