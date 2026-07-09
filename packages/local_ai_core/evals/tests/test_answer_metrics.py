import pytest

from local_ai_core.evals.answer_metrics import (
    keyword_overlap_relevance,
    must_contain_score,
    must_not_contain_score,
    refusal_check,
)


class TestMustContainScore:
    def test_full_score_when_all_phrases_present(self):
        assert must_contain_score("The link expires in 15 minutes.", ["15 minutes"]) == 1.0

    def test_partial_score_when_some_phrases_missing(self):
        score = must_contain_score("The link expires in 15 minutes.", ["15 minutes", "backup codes"])
        assert score == pytest.approx(0.5)

    def test_case_insensitive_matching(self):
        assert must_contain_score("Fifteen Minutes later.", ["fifteen minutes"]) == 1.0

    def test_no_requirements_is_vacuously_full_score(self):
        assert must_contain_score("anything", []) == 1.0


class TestMustNotContainScore:
    def test_full_score_when_forbidden_phrase_absent(self):
        assert must_not_contain_score("The link expires in 15 minutes.", ["30 days"]) == 1.0

    def test_zero_score_when_forbidden_phrase_present(self):
        assert must_not_contain_score("Data is deleted automatically.", ["deleted automatically"]) == 0.0

    def test_no_requirements_is_vacuously_full_score(self):
        assert must_not_contain_score("anything", []) == 1.0


class TestKeywordOverlapRelevance:
    def test_shared_words_score_above_zero(self):
        assert keyword_overlap_relevance("how long is the reset link valid", "the reset link is valid for 15 minutes") > 0.0

    def test_completely_unrelated_answer_scores_zero(self):
        assert keyword_overlap_relevance("how long is the reset link valid", "distant galaxies contain stars") == 0.0

    def test_empty_question_scores_zero(self):
        assert keyword_overlap_relevance("", "some answer") == 0.0


class TestRefusalCheck:
    def test_recognizes_the_default_refusal_phrase(self):
        assert refusal_check("I don't know based on the provided documents.") is True

    def test_confident_wrong_answer_is_not_a_refusal(self):
        assert refusal_check("The CEO of Nimbus is Jane Smith.") is False

    def test_custom_refusal_phrases_are_recognized(self):
        assert refusal_check("Unable to answer this question.", refusal_phrases=("unable to answer",)) is True
