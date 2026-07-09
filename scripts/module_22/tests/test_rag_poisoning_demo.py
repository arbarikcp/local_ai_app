import rag_poisoning_demo as sut


class TestRunLab:
    def test_the_malicious_document_is_quarantined(self):
        result = sut.run_lab()
        assert result["malicious_allowed"] is False
        assert len(result["malicious_flagged_patterns"]) > 0

    def test_the_clean_document_is_allowed(self):
        result = sut.run_lab()
        assert result["clean_allowed"] is True


class TestResultToMarkdown:
    def test_markdown_reports_both_outcomes(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "allowed: False" in markdown
        assert "allowed: True" in markdown
