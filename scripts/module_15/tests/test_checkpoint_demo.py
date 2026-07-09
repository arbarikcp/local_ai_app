import checkpoint_demo as sut


class TestRunLab:
    async def test_first_run_fails_at_categorize(self):
        result = await sut.run_lab()
        assert result["first_run_stopped_reason"] == "failed"
        assert result["first_run_final_node"] == "categorize"

    async def test_first_run_checkpointed_the_fetch_result(self):
        result = await sut.run_lab()
        assert result["first_run_state"]["tickets_fetched"] == 5

    async def test_resumed_run_reaches_the_end(self):
        result = await sut.run_lab()
        assert result["resumed_run_stopped_reason"] == "end"

    async def test_resumed_run_state_includes_both_old_and_new_progress(self):
        result = await sut.run_lab()
        state = result["resumed_run_final_state"]
        assert state["tickets_fetched"] == 5  # from before the restart
        assert state["categorized"] is True  # from after the restart
        assert state["done"] is True


class TestResultToMarkdown:
    async def test_includes_both_runs_outcomes(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "categorize" in markdown
        assert "end" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "checkpoint" in captured.out.lower()
