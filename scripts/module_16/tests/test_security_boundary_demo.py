import security_boundary_demo as sut


class TestRunLab:
    async def test_discovery_is_not_authorization(self):
        result = await sut.run_lab()
        assert result["guest_sees_sql_query_in_tools_list"] is True
        assert result["guest_call_denied"] is True

    async def test_dangerous_tool_denied_without_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["denied_write_no_approval_gate"] is True

    async def test_dangerous_tool_succeeds_with_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["approved_write_succeeded"] is True
        assert result["file_actually_written"] is True

    async def test_malicious_tool_description_is_flagged(self):
        result = await sut.run_lab()
        assert len(result["malicious_tool_description_flagged"]) > 0

    async def test_llm_summary_is_produced_from_the_real_tool_result(self):
        result = await sut.run_lab()
        assert "2 open tickets" in result["llm_summary"]

    async def test_every_dispatch_call_is_audit_logged(self):
        result = await sut.run_lab()
        assert result["total_audit_entries"] > 0


class TestResultToMarkdown:
    async def test_includes_the_security_boundary_facts(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "discovery is not authorization" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "audit log" in captured.out.lower()
