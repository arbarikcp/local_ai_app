import approval_and_dangerous_tools_demo as sut


class TestRunLab:
    async def test_denied_by_default_when_no_real_approval_gate_is_configured(self):
        result = await sut.run_lab()
        assert result["denied_by_default_null_gate"] is True

    async def test_approved_write_succeeds(self):
        result = await sut.run_lab()
        assert result["approved_write_succeeded"] is True

    async def test_denied_write_fails(self):
        result = await sut.run_lab()
        assert result["denied_write_by_callback"] is True

    async def test_permission_denial_happens_before_approval(self):
        result = await sut.run_lab()
        assert result["permission_denied_before_approval"] is True

    async def test_budget_allows_exactly_two_calls_then_denies(self):
        result = await sut.run_lab()
        assert result["budget_audit_outcomes"] == ["success", "success", "denied"]

    async def test_only_approved_files_are_actually_written(self):
        result = await sut.run_lab()
        assert "secrets.txt" not in result["sandbox_files_written"]
        assert "notes.txt" in result["sandbox_files_written"]

    async def test_every_attempt_is_audit_logged(self):
        result = await sut.run_lab()
        # denied_by_default(1) + approved(1) + denied_by_callback(1) + permission_denied(1) + budget(3) = 7
        assert result["total_audit_entries"] == 7


class TestResultToMarkdown:
    async def test_includes_the_budget_outcomes(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "success" in markdown
        assert "denied" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "audit log" in captured.out.lower()
