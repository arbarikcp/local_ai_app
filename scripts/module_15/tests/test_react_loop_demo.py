import react_loop_demo as sut


class TestRunLab:
    async def test_happy_path_reaches_a_final_answer(self):
        result = await sut.run_lab()
        assert result["happy_stopped_reason"] == "final_answer"
        assert "3" in result["happy_final_answer"]

    async def test_adversarial_prompt_trips_the_loop_guard(self):
        result = await sut.run_lab()
        assert result["adversarial_stopped_reason"] == "loop_detected"

    async def test_loop_guard_stops_it_well_before_the_safety_budget_would(self):
        result = await sut.run_lab()
        assert result["adversarial_would_have_looped_forever"] is True


class TestResultToMarkdown:
    async def test_includes_both_labs(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Lab 1" in markdown
        assert "Lab 2" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "adversarial" in captured.out.lower()
