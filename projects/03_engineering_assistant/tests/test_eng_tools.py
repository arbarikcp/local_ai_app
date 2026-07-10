import shutil
from pathlib import Path

import pytest

from eng_patch_guard import PatchScopeError
from eng_tools import ApplyPatchArgs, ProposePatchArgs, make_apply_patch_tool, make_propose_patch_tool, make_run_tests_tool
from local_ai_agents.tools.base import ToolCallProposal
from local_ai_agents.tools.patch_tools import PatchFormatError
from local_ai_agents.tools.registry import ToolRegistry
from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.policies.approval import AutoApprovalGate
from local_ai_core.runtimes.fake import FakeRuntime

DEMO_REPO = Path(__file__).resolve().parent.parent / "demo_repo"

VALID_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,3 +20,5 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""


def make_sandbox(tmp_path) -> Path:
    sandbox = tmp_path / "repo"
    shutil.copytree(DEMO_REPO, sandbox)
    return sandbox


class TestProposePatchTool:
    async def test_calls_the_real_runtime_and_returns_its_text(self):
        runtime = FakeRuntime(default_response=VALID_PATCH)
        tool = make_propose_patch_tool(runtime, "fake-model")
        result = await tool.handler(
            ProposePatchArgs(instruction="fix the bug", file_contents={"inventory/stock.py": "..."})
        )
        assert result == VALID_PATCH.strip()  # propose_patch() strips trailing whitespace


class TestApplyPatchTool:
    async def test_a_valid_patch_for_the_expected_file_is_applied(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        tool = make_apply_patch_tool(sandbox)
        result = await tool.handler(ApplyPatchArgs(patch_text=VALID_PATCH, expected_file_path="inventory/stock.py"))
        assert result == "inventory/stock.py"
        assert "cannot remove more than current quantity" in (sandbox / "inventory" / "stock.py").read_text()

    async def test_a_patch_for_the_wrong_file_is_rejected_before_any_write(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        tool = make_apply_patch_tool(sandbox)
        original_content = (sandbox / "inventory" / "stock.py").read_text()
        with pytest.raises(PatchScopeError):
            await tool.handler(ApplyPatchArgs(patch_text=VALID_PATCH, expected_file_path="inventory/pricing.py"))
        assert (sandbox / "inventory" / "stock.py").read_text() == original_content

    async def test_a_malformed_patch_is_rejected(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        tool = make_apply_patch_tool(sandbox)
        with pytest.raises(PatchFormatError):
            await tool.handler(ApplyPatchArgs(patch_text="not a real patch", expected_file_path="inventory/stock.py"))

    async def test_is_dangerous_and_marked_for_approval(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        tool = make_apply_patch_tool(sandbox)
        assert tool.dangerous is True


class TestRunTestsTool:
    async def test_running_the_real_demo_repo_tests_reports_the_real_failure(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        tool = make_run_tests_tool(sandbox)
        result = await tool.handler(tool.args_model(test_path="tests"))
        assert result["passed"] is False
        assert "1 failed" in result["stdout"] or "failed" in result["stdout"]

    async def test_running_tests_after_a_real_patch_fix_passes(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        apply_tool = make_apply_patch_tool(sandbox)
        await apply_tool.handler(ApplyPatchArgs(patch_text=VALID_PATCH, expected_file_path="inventory/stock.py"))

        run_tool = make_run_tests_tool(sandbox)
        result = await run_tool.handler(run_tool.args_model(test_path="tests"))
        assert result["passed"] is True


class TestToolsRouteThroughToolExecutor:
    async def test_apply_patch_via_tool_executor_is_audit_logged_and_approval_gated(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        registry = ToolRegistry()
        registry.register(make_apply_patch_tool(sandbox))
        executor = ToolExecutor(registry, approval_gate=AutoApprovalGate())

        proposal = ToolCallProposal(
            tool_name="apply_patch",
            raw_arguments={"patch_text": VALID_PATCH, "expected_file_path": "inventory/stock.py"},
        )
        result = await executor.execute(proposal, trace_id="trace-1")

        assert result.success is True
        assert "cannot remove more than current quantity" in (sandbox / "inventory" / "stock.py").read_text()
