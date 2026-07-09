import tool_registry_demo as sut


class TestRunLab:
    async def test_registers_all_three_tools(self):
        result = await sut.run_lab()
        assert set(result["registered_tools"]) == {"calculator", "search_files", "sql_query"}

    async def test_calculator_computes_correctly(self):
        result = await sut.run_lab()
        assert result["calculator_result"] == "20"

    async def test_file_search_finds_the_password_reset_document(self):
        result = await sut.run_lab()
        assert "password_reset.md" in result["file_search_result"]

    async def test_sql_select_returns_only_open_tickets(self):
        result = await sut.run_lab()
        assert len(result["sql_select_result"]) == 2

    async def test_sql_write_attempt_is_denied(self):
        result = await sut.run_lab()
        assert result["sql_write_attempt_denied"] is True

    async def test_llm_proposed_tool_call_executes_correctly(self):
        result = await sut.run_lab()
        assert result["llm_proposed_tool"] == "calculator"
        assert result["llm_proposed_result"] == "144"


class TestResultToMarkdown:
    async def test_includes_registered_tools(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "calculator" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "tool registry" in captured.out
