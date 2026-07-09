import index_repo_demo as sut


class TestRunLab:
    async def test_indexes_both_files(self):
        result = await sut.run_lab()
        assert set(result["files_indexed"]) == {"calculator.py", "tests/test_calculator.py"}

    async def test_calculator_functions_match_the_real_source(self):
        result = await sut.run_lab()
        assert result["calculator_functions"] == ["add", "subtract", "multiply", "divide", "average"]

    async def test_repo_map_reports_the_real_line_number_for_average(self):
        result = await sut.run_lab()
        average_symbol = next(s for s in result["repo_map"]["calculator.py"] if s["name"] == "average")
        assert average_symbol["line"] == 22

    async def test_search_finds_valueerror_in_both_files(self):
        result = await sut.run_lab()
        assert len(result["search_matches"]) == 2


class TestResultToMarkdown:
    async def test_includes_the_repo_map(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "average" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "repo index" in captured.out.lower()
