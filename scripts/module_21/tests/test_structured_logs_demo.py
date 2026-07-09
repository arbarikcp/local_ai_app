import structured_logs_demo as sut


class TestRunLab:
    def test_generates_a_real_trace_id(self):
        result = sut.run_lab()
        assert len(result["trace_id"]) > 0

    def test_full_policy_includes_the_raw_email(self):
        result = sut.run_lab()
        assert "jane.doe@example.com" in result["fields_by_policy"]["full"]["prompt"]

    def test_redacted_policy_removes_the_email(self):
        result = sut.run_lab()
        assert "jane.doe@example.com" not in result["fields_by_policy"]["redacted"]["prompt"]
        assert "[EMAIL]" in result["fields_by_policy"]["redacted"]["prompt"]

    def test_hash_only_and_none_policies_never_include_the_raw_prompt(self):
        result = sut.run_lab()
        assert "prompt" not in result["fields_by_policy"]["hash_only"]
        assert "prompt" not in result["fields_by_policy"]["none"]


class TestResultToMarkdown:
    def test_markdown_lists_every_policy(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        for policy in ["full", "redacted", "hash_only", "none"]:
            assert policy in markdown
