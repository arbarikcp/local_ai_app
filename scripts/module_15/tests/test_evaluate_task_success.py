import evaluate_task_success as sut


class TestRunLab:
    async def test_evaluates_every_golden_case(self):
        rows = await sut.run_lab()
        assert len(rows) == len(sut.GOLDEN_CASES)

    async def test_correct_expectations_succeed(self):
        rows = await sut.run_lab()
        normal_run = next(r for r in rows if r["case_id"] == "normal_run")
        assert normal_run["success"] is True

    async def test_a_deliberately_wrong_expectation_fails(self):
        rows = await sut.run_lab()
        wrong = next(r for r in rows if r["case_id"] == "deliberately_wrong_expectation")
        assert wrong["success"] is False

    async def test_all_cases_reach_the_correct_real_ticket_count_regardless_of_scoring(self):
        rows = await sut.run_lab()
        assert all(r["open_ticket_count"] == 3 for r in rows)


class TestResultToMarkdown:
    async def test_reports_the_success_rate(self):
        rows = await sut.run_lab()
        markdown = sut.result_to_markdown(rows)
        assert "2/3" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "success rate" in captured.out.lower()
