import workflow_graph_demo as sut


class TestRunLab:
    async def test_dangerous_node_denied_without_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["denied_stopped_reason"] == "approval_denied"

    async def test_dangerous_node_succeeds_with_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["approved_stopped_reason"] == "end"
        assert "3 open tickets" in result["approved_summary"]

    async def test_the_file_is_only_written_after_approval(self):
        result = await sut.run_lab()
        assert result["summary_file_written"] is True

    async def test_the_adversarial_runtime_cannot_provoke_a_loop(self):
        result = await sut.run_lab()
        assert result["immune_stopped_reason"] == "end"
        assert result["immune_llm_calls_made"] == 1


class TestResultToMarkdown:
    async def test_includes_the_request(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert result["request"] in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "immune" in captured.out.lower()
