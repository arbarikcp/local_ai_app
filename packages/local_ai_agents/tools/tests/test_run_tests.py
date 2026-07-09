import pytest

from local_ai_agents.tools.run_tests import RunTestsArgs, RunTestsTimeoutError, make_run_tests_tool, run_tests


def make_passing_repo(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    return tmp_path


def make_failing_repo(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_broken.py").write_text("def test_broken():\n    assert 1 + 1 == 3\n")
    return tmp_path


def make_slow_repo(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(2)\n")
    return tmp_path


class TestRunTests:
    async def test_a_passing_suite_reports_passed(self, tmp_path):
        make_passing_repo(tmp_path)
        result = await run_tests(tmp_path)
        assert result.passed is True
        assert result.exit_code == 0

    async def test_a_failing_suite_reports_not_passed(self, tmp_path):
        make_failing_repo(tmp_path)
        result = await run_tests(tmp_path)
        assert result.passed is False
        assert result.exit_code != 0

    async def test_stdout_contains_real_pytest_output(self, tmp_path):
        make_failing_repo(tmp_path)
        result = await run_tests(tmp_path)
        assert "test_broken" in result.stdout

    async def test_a_slow_test_exceeding_the_timeout_raises(self, tmp_path):
        make_slow_repo(tmp_path)
        with pytest.raises(RunTestsTimeoutError):
            await run_tests(tmp_path, timeout_seconds=0.2)


class TestMakeRunTestsTool:
    def test_tool_is_dangerous(self, tmp_path):
        tool = make_run_tests_tool(tmp_path)
        assert tool.dangerous is True

    async def test_handler_returns_a_serializable_result(self, tmp_path):
        make_passing_repo(tmp_path)
        tool = make_run_tests_tool(tmp_path)
        result = await tool.handler(RunTestsArgs())
        assert result["passed"] is True
