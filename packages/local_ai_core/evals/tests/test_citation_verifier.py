from local_ai_core.evals.citation_verifier import citation_faithfulness_score, citations_are_grounded


class TestCitationsAreGrounded:
    def test_true_when_every_citation_was_retrieved(self):
        assert citations_are_grounded(["a::0", "b::0"], ["a::0", "b::0", "c::0"]) is True

    def test_false_when_a_citation_was_never_retrieved(self):
        assert citations_are_grounded(["a::0", "invented::9"], ["a::0", "b::0"]) is False

    def test_no_citations_is_vacuously_grounded(self):
        assert citations_are_grounded([], ["a::0"]) is True


class TestCitationFaithfulnessScore:
    def test_no_citations_is_vacuously_faithful(self):
        assert citation_faithfulness_score("No citations here.", {}) == 1.0

    def test_high_score_when_sentence_overlaps_the_cited_chunk(self):
        answer = "The password reset link expires in 15 minutes [password_reset::0]."
        chunks = {"password_reset::0": "The password reset link expires in fifteen minutes for security."}
        score = citation_faithfulness_score(answer, chunks)
        assert score > 0.5

    def test_zero_score_when_sentence_shares_no_words_with_the_cited_chunk(self):
        answer = "Distant galaxies contain billions of stars [password_reset::0]."
        chunks = {"password_reset::0": "The password reset link expires in fifteen minutes."}
        score = citation_faithfulness_score(answer, chunks)
        assert score == 0.0

    def test_citation_pointing_to_a_missing_chunk_scores_zero(self):
        answer = "Something happened [does_not_exist::0]."
        score = citation_faithfulness_score(answer, {})
        assert score == 0.0

    def test_averages_across_multiple_citations(self):
        answer = "Reset link is 15 minutes [a::0]. Distant galaxies exist [b::0]."
        chunks = {
            "a::0": "The password reset link expires in fifteen minutes.",
            "b::0": "Billing is charged monthly.",
        }
        score = citation_faithfulness_score(answer, chunks)
        assert 0.0 < score < 1.0
