"""run_tests tool (curriculum's required tools list: "run_tests(command)
with approval or sandbox") - a real `subprocess` call to `pytest` inside
the sandboxed repo directory, with a real timeout and real captured
stdout/exit code, not a simulated pass/fail. Marked `dangerous=True` -
executing code is exactly the kind of action curriculum's dangerous-tools
list names, even when the "code" is just the repo's own test suite.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool


class RunTestsArgs(BaseModel):
    test_path: str = Field(default="tests")
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)


class RunTestsTimeoutError(Exception):
    pass


@dataclass(frozen=True)
class TestRunResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str


def _run_pytest_subprocess(repo_dir: Path, test_path: str, timeout_seconds: float) -> TestRunResult:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, sandboxed cwd
            [sys.executable, "-m", "pytest", test_path, "-q", "--color=no"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunTestsTimeoutError(f"Test run exceeded the {timeout_seconds}s timeout") from exc

    return TestRunResult(
        passed=completed.returncode == 0, exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
    )


async def run_tests(repo_dir: Path, test_path: str = "tests", timeout_seconds: float = 30.0) -> TestRunResult:
    return await asyncio.to_thread(_run_pytest_subprocess, repo_dir, test_path, timeout_seconds)


def make_run_tests_tool(repo_dir: Path) -> Tool:
    async def handler(args: RunTestsArgs) -> dict:
        result = await run_tests(repo_dir, args.test_path, args.timeout_seconds)
        return {"passed": result.passed, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    return Tool(
        name="run_tests",
        description="Run the repo's pytest suite in a sandboxed subprocess. Requires approval.",
        args_model=RunTestsArgs,
        handler=handler,
        dangerous=True,
    )
