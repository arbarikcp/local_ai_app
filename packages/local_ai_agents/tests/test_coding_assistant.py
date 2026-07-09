import shutil
from pathlib import Path

from local_ai_agents.coding_assistant import build_coding_assistant_graph
from local_ai_agents.executors.workflow_executor import WorkflowExecutor
from local_ai_agents.planners.safety_budget import AgentSafetyBudget
from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate
from local_ai_core.runtimes.fake import FakeRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


def copy_sample_repo(tmp_path) -> Path:
    dest = tmp_path / "mini_calculator"
    shutil.copytree(SAMPLE_REPO, dest)
    return dest


def make_budget() -> AgentSafetyBudget:
    return AgentSafetyBudget(max_steps=10, max_tool_calls=10, max_runtime_seconds=60, max_tokens_total=8000)


class TestHappyPath:
    async def test_the_real_pre_existing_bug_is_fixed_and_tests_pass(self, tmp_path):
        repo_dir = copy_sample_repo(tmp_path)
        runtime = FakeRuntime(default_response=VALID_PATCH)
        graph = build_coding_assistant_graph(repo_dir, runtime, model="fake-model")
        executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())

        result = await executor.run(
            {"query": "average", "instruction": "fix average() to return 0 for an empty list"}, make_budget()
        )

        assert result.stopped_reason == "end"
        assert result.final_state["patch_valid"] is True
        assert result.final_state["tests_passed"] is True

    async def test_the_file_on_disk_actually_changed(self, tmp_path):
        repo_dir = copy_sample_repo(tmp_path)
        runtime = FakeRuntime(default_response=VALID_PATCH)
        graph = build_coding_assistant_graph(repo_dir, runtime, model="fake-model")
        executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())

        await executor.run({"query": "average", "instruction": "fix average"}, make_budget())

        content = (repo_dir / "calculator.py").read_text()
        assert "if not numbers:" in content


class TestApprovalRequired:
    async def test_apply_is_denied_without_a_real_approval_gate(self, tmp_path):
        repo_dir = copy_sample_repo(tmp_path)
        runtime = FakeRuntime(default_response=VALID_PATCH)
        graph = build_coding_assistant_graph(repo_dir, runtime, model="fake-model")
        executor = WorkflowExecutor(graph, approval_gate=NullApprovalGate())

        result = await executor.run({"query": "average", "instruction": "fix average"}, make_budget())

        assert result.stopped_reason == "approval_denied"
        # The file was never touched - approval was denied before apply_patch ran.
        original = (SAMPLE_REPO / "calculator.py").read_text()
        assert (repo_dir / "calculator.py").read_text() == original


class TestInvalidPatch:
    async def test_an_invalid_patch_routes_to_report_without_attempting_to_apply(self, tmp_path):
        repo_dir = copy_sample_repo(tmp_path)
        runtime = FakeRuntime(default_response="this is not a valid unified diff")
        graph = build_coding_assistant_graph(repo_dir, runtime, model="fake-model")
        executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())

        result = await executor.run({"query": "average", "instruction": "fix average"}, make_budget())

        assert result.stopped_reason == "end"
        assert result.final_state["patch_valid"] is False
        assert "tests_passed" not in result.final_state  # run_tests never reached
        original = (SAMPLE_REPO / "calculator.py").read_text()
        assert (repo_dir / "calculator.py").read_text() == original


class TestNoSearchMatches:
    async def test_no_matches_routes_directly_to_report(self, tmp_path):
        repo_dir = copy_sample_repo(tmp_path)
        runtime = FakeRuntime(default_response=VALID_PATCH)
        graph = build_coding_assistant_graph(repo_dir, runtime, model="fake-model")
        executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())

        result = await executor.run(
            {"query": "this_string_appears_nowhere_xyz", "instruction": "fix something"}, make_budget()
        )

        assert result.stopped_reason == "end"
        assert result.final_state["target_file"] is None
