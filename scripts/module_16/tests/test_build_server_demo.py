import build_server_demo as sut


class TestRunLab:
    async def test_registers_both_tools(self):
        result = await sut.run_lab()
        assert set(result["tool_names"]) == {"search_files", "sql_query"}

    async def test_tool_metadata_includes_a_flagged_patterns_field(self):
        result = await sut.run_lab()
        assert "flagged_patterns" in result["tool_metadata_example"]

    async def test_file_search_finds_the_password_reset_document(self):
        result = await sut.run_lab()
        assert "password_reset.md" in result["search_result"]

    async def test_sql_query_returns_the_open_ticket_count(self):
        result = await sut.run_lab()
        assert result["sql_result"] == [{"n": 2}]

    async def test_resource_is_registered_and_readable(self):
        result = await sut.run_lab()
        assert "password_reset.md" in result["resource_uris"]
        assert result["resource_content_length"] > 0

    async def test_prompt_renders_with_real_arguments(self):
        result = await sut.run_lab()
        assert "How long is a reset link valid?" in result["rendered_prompt"]


class TestResultToMarkdown:
    async def test_includes_registered_tools(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "search_files" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "MCP-like server" in captured.out
