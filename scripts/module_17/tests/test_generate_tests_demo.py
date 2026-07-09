import generate_tests_demo as sut


class TestRunLab:
    async def test_the_generated_test_increases_the_passing_count_by_one(self):
        result = await sut.run_lab()
        assert result["passed_count_after"] == result["passed_count_before"] + 1

    async def test_the_generated_test_source_is_included(self):
        result = await sut.run_lab()
        assert "test_subtract_generated" in result["generated_test_source"]

    async def test_six_tests_pass_before_generation(self):
        result = await sut.run_lab()
        assert result["passed_count_before"] == 6


class TestResultToMarkdown:
    async def test_includes_the_generated_source(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "test_subtract_generated" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "generate a test" in captured.out.lower()
