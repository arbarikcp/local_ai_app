import patch_and_test_demo as sut


class TestRunLab:
    async def test_the_repo_genuinely_fails_before_any_patch(self):
        result = await sut.run_lab()
        assert result["before_patch_passed"] is False
        assert "1 failed" in result["before_patch_stdout_tail"]

    async def test_apply_is_denied_without_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["denied_stopped_reason"] == "approval_denied"

    async def test_apply_succeeds_with_a_real_approval_gate(self):
        result = await sut.run_lab()
        assert result["approved_stopped_reason"] == "end"

    async def test_the_repo_genuinely_passes_after_the_patch(self):
        result = await sut.run_lab()
        assert result["approved_tests_passed"] is True
        assert "failed" not in result["approved_stdout_tail"]

    async def test_a_hallucinated_patch_is_rejected(self):
        result = await sut.run_lab()
        assert result["hallucinated_patch_rejected"] is True

    async def test_the_file_is_untouched_after_a_rejected_hallucinated_patch(self):
        result = await sut.run_lab()
        assert result["hallucinated_patch_file_untouched"] is True


class TestResultToMarkdown:
    async def test_includes_the_before_and_after_test_output(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Before the patch" in markdown
        assert "After the patch" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "hallucinated" in captured.out.lower()
