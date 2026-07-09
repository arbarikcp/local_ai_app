import citation_and_injection_checks as sut


class TestRunLab:
    async def test_a_well_grounded_answer_scores_highly_faithful(self):
        result = await sut.run_lab()
        assert result["good_case_faithfulness_score"] > 0.5

    async def test_a_deliberately_unfaithful_answer_scores_lower(self):
        result = await sut.run_lab()
        assert result["unfaithful_answer_faithfulness_score"] < result["good_case_faithfulness_score"]

    async def test_the_malicious_document_is_flagged(self):
        result = await sut.run_lab()
        assert len(result["malicious_document_patterns_matched"]) > 0

    async def test_the_clean_document_is_not_flagged(self):
        result = await sut.run_lab()
        assert result["clean_document_patterns_matched"] == []


class TestResultToMarkdown:
    async def test_includes_the_matched_patterns(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "injection patterns matched" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "citation verifier" in captured.out
