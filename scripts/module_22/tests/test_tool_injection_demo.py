import tool_injection_demo as sut


class TestRunLab:
    async def test_the_injected_dangerous_call_is_denied(self):
        result = await sut.run_lab()
        assert result["injected_call_succeeded"] is False
        assert "not permitted" in result["injected_call_error"]

    async def test_the_legitimate_approved_call_succeeds(self):
        result = await sut.run_lab()
        assert result["legitimate_call_succeeded"] is True

    async def test_both_attempts_are_audit_logged(self):
        result = await sut.run_lab()
        assert result["audit_entry_count"] == 2
        assert result["audit_outcomes"] == ["denied", "success"]


class TestResultToMarkdown:
    async def test_markdown_reports_the_denial(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "succeeded=False" in markdown
        assert "succeeded=True" in markdown
