"""Labs 4-7 - propose a patch, validate it (including a rejected
hallucinated patch), apply it with human approval, and run tests: a real
failure before the patch, a real pass after. The strongest "real proof"
in this course - not a staged before/after, an actual pytest run against
an actual pre-existing bug in `datasets/code_repos/mini_calculator/`.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_agents.coding_assistant import build_coding_assistant_graph  # noqa: E402
from local_ai_agents.executors.workflow_executor import WorkflowExecutor  # noqa: E402
from local_ai_agents.planners.safety_budget import AgentSafetyBudget  # noqa: E402
from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate  # noqa: E402
from local_ai_agents.tools.patch_tools import PatchFormatError, apply_patch  # noqa: E402
from local_ai_agents.tools.run_tests import run_tests  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_REPO = REPO_ROOT / "datasets" / "code_repos" / "mini_calculator"

VALID_PATCH = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -22,2 +22,4 @@\n"
    " def average(numbers: list[float]) -> float:\n"
    "-    return sum(numbers) / len(numbers)\n"
    "+    if not numbers:\n"
    "+        return 0.0\n"
    "+    return sum(numbers) / len(numbers)\n"
)

# Same line number, plausible-looking, but describes code that doesn't
# actually exist in the file - a real stand-in for a hallucinated patch
# (theory doc §13, "Code hallucination").
HALLUCINATED_PATCH = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -22,2 +22,3 @@\n"
    " def average(numbers: list[float]) -> float:\n"
    "-    return statistics.mean(numbers)\n"
    "+    if not numbers:\n"
    "+        return 0.0\n"
)


def copy_sample_repo(tmp_dir: Path) -> Path:
    dest = tmp_dir / "mini_calculator"
    shutil.copytree(SAMPLE_REPO, dest)
    return dest


def make_budget() -> AgentSafetyBudget:
    return AgentSafetyBudget(max_steps=10, max_tool_calls=10, max_runtime_seconds=60, max_tokens_total=8000)


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # --- Lab 6 (part 1): the real pre-existing failure, before any patch.
        before_repo = copy_sample_repo(tmp_dir / "before")
        before_result = await run_tests(before_repo)

        # --- Labs 4-7: propose, validate, approve, apply, re-run tests.
        patched_repo = copy_sample_repo(tmp_dir / "patched")
        runtime = FakeRuntime(default_response=VALID_PATCH)
        graph = build_coding_assistant_graph(patched_repo, runtime, model="fake-model")

        # Lab 7a: no real approval gate -> denied, nothing applied.
        denied_executor = WorkflowExecutor(graph, approval_gate=NullApprovalGate())
        denied_result = await denied_executor.run(
            {"query": "average", "instruction": "fix average() to return 0 for an empty list"}, make_budget()
        )

        # Lab 7b: a real approval gate -> the patch is applied and tests re-run.
        approved_executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())
        approved_result = await approved_executor.run(
            {"query": "average", "instruction": "fix average() to return 0 for an empty list"}, make_budget()
        )

        # --- Lab 5: a deliberately hallucinated patch is rejected, not misapplied.
        hallucination_repo = copy_sample_repo(tmp_dir / "hallucination")
        hallucination_rejected = False
        try:
            apply_patch(hallucination_repo, HALLUCINATED_PATCH)
        except PatchFormatError:
            hallucination_rejected = True
        hallucination_file_untouched = (hallucination_repo / "calculator.py").read_text() == (
            SAMPLE_REPO / "calculator.py"
        ).read_text()

        return {
            "before_patch_passed": before_result.passed,
            "before_patch_stdout_tail": before_result.stdout.strip().splitlines()[-1],
            "denied_stopped_reason": denied_result.stopped_reason,
            "approved_stopped_reason": approved_result.stopped_reason,
            "approved_tests_passed": approved_result.final_state.get("tests_passed"),
            "approved_stdout_tail": approved_result.final_state.get("test_stdout", "").strip().splitlines()[-1]
            if approved_result.final_state.get("test_stdout")
            else "",
            "hallucinated_patch_rejected": hallucination_rejected,
            "hallucinated_patch_file_untouched": hallucination_file_untouched,
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 4-7 - propose, validate, approve, apply a real patch; run real tests before and after\n\n"
        f"- **Before the patch**: tests passed = {result['before_patch_passed']} "
        f"-> `{result['before_patch_stdout_tail']}`\n"
        f"- Apply denied with no real approval gate: {result['denied_stopped_reason']}\n"
        f"- Apply approved with a real approval gate: {result['approved_stopped_reason']}\n"
        f"- **After the patch**: tests passed = {result['approved_tests_passed']} "
        f"-> `{result['approved_stdout_tail']}`\n"
        f"- Hallucinated patch rejected (context mismatch): {result['hallucinated_patch_rejected']}\n"
        f"- File untouched after the rejected hallucinated patch: {result['hallucinated_patch_file_untouched']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
