"""Tool registrations (ARCHITECTURE.md "Tool registrations") — closes the
gap confirmed by survey: `patch_tools.py`'s `propose_patch`/`apply_patch`
have no `Tool`/registry wrappers, so calling them bypasses `ToolExecutor`'s
audit-logging/permission/approval/budget layer entirely. Also applies
Module 22's `with_timeout()` uniformly (previously only `run_tests`'s own
bespoke `subprocess` timeout existed anywhere in this pipeline).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.patch_tools import apply_patch, propose_patch, validate_patch_format
from local_ai_agents.tools.run_tests import RunTestsArgs, TestRunResult, run_tests
from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.security.tool_call_timeout import with_timeout

from eng_command_safety import validate_test_command
from eng_patch_guard import validate_hunk_line_counts, validate_patch_scope

_APPLY_PATCH_TIMEOUT_SECONDS = 10.0


class ProposePatchArgs(BaseModel):
    instruction: str
    file_contents: dict[str, str]


def make_propose_patch_tool(runtime: LLMRuntime, model: str) -> Tool:
    async def handler(args: ProposePatchArgs) -> str:
        return await propose_patch(args.instruction, args.file_contents, runtime, model)

    return Tool(
        name="propose_patch",
        description="Propose a code fix as a unified diff, given an instruction and relevant file contents.",
        args_model=ProposePatchArgs,
        handler=handler,
    )


class ApplyPatchArgs(BaseModel):
    patch_text: str
    expected_file_path: str


def make_apply_patch_tool(allowed_base: Path) -> Tool:
    """Runs every validation this project adds (`validate_patch_scope`,
    `validate_hunk_line_counts`) in addition to Module 17's own
    `validate_patch_format`/context-mismatch check inside `apply_patch`
    itself - all three must pass before a single byte is written.
    """

    async def handler(args: ApplyPatchArgs) -> str:
        parsed = validate_patch_format(args.patch_text)
        validate_patch_scope(parsed, args.expected_file_path)
        validate_hunk_line_counts(args.patch_text)

        async def _apply() -> str:
            return await asyncio.to_thread(apply_patch, allowed_base, args.patch_text)

        return await with_timeout(_apply, timeout_seconds=_APPLY_PATCH_TIMEOUT_SECONDS)

    return Tool(
        name="apply_patch",
        description="Apply a validated unified diff to a sandboxed file. Requires approval.",
        args_model=ApplyPatchArgs,
        handler=handler,
        dangerous=True,
    )


def make_run_tests_tool(repo_dir: Path) -> Tool:
    """Builds the exact fixed argv `run_tests()` would use internally and
    validates it via `eng_command_safety.validate_test_command()` before
    ever running it - real, even though this particular pathway never
    varies, so a future model-suggested command variant has a real check
    already wired in and tested, not one added after the fact.
    """

    async def handler(args: RunTestsArgs) -> dict:
        argv = [sys.executable, "-m", "pytest", args.test_path, "-q", "--color=no"]
        validate_test_command(argv)

        async def _run() -> TestRunResult:
            return await run_tests(repo_dir, args.test_path, args.timeout_seconds)

        result = await with_timeout(_run, timeout_seconds=args.timeout_seconds + 5.0)
        return {"passed": result.passed, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    return Tool(
        name="run_tests",
        description="Run the repo's pytest suite in a sandboxed subprocess. Requires approval.",
        args_model=RunTestsArgs,
        handler=handler,
        dangerous=True,
    )
