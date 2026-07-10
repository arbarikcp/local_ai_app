import config_and_registry_demo as sut


class TestRunLab:
    def test_loads_the_real_committed_config(self):
        result = sut.run_lab()
        assert result["default_chat_model"] == "llama3.2:3b"
        assert result["max_concurrent_requests"] == 1
        assert result["redact_pii_in_logs"] is True

    def test_validation_rejects_a_bad_config(self):
        result = sut.run_lab()
        assert result["validation_rejected_bad_input"] is True

    def test_parses_the_real_committed_catalog(self):
        result = sut.run_lab()
        assert result["registry_size"] == 10
        assert result["categories"] == ["chat", "code", "embedding", "reranker"]
        assert len(result["chat_models"]) == 5


class TestResultToMarkdown:
    def test_markdown_reports_the_registry_size(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "10 entries" in markdown
