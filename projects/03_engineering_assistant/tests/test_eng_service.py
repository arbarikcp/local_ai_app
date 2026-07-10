import shutil
from pathlib import Path

from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate
from local_ai_agents.tools.base import ToolCallProposal
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime

from eng_service import build_eng_context

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"
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


def make_config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "a",
                "default_extraction": "b",
                "default_code": "test-code-model",
                "default_embedding": "d",
            },
        }
    )


def make_sandbox(tmp_path) -> Path:
    sandbox = tmp_path / "repo"
    shutil.copytree(DEMO_REPO, sandbox)
    return sandbox


class TestBuildEngContext:
    def test_registers_every_capability(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_eng_context(config, model_catalog_path=REPO_ROOT_CATALOG, repo_dir=make_sandbox(tmp_path))
        registered = {t.name for t in ctx.tool_registry.list_tools()}
        assert registered == {"list_symbols", "search_repo", "read_file", "propose_patch", "apply_patch", "run_tests"}

    def test_defaults_to_a_fail_closed_approval_gate(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_eng_context(config, model_catalog_path=REPO_ROOT_CATALOG, repo_dir=make_sandbox(tmp_path))
        assert isinstance(ctx.tool_executor._approval_gate, NullApprovalGate)


class TestReadOnlyToolsRunWithoutApproval:
    async def test_search_repo_needs_no_approval(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_eng_context(config, model_catalog_path=REPO_ROOT_CATALOG, repo_dir=make_sandbox(tmp_path))
        result = await ctx.tool_executor.execute(
            ToolCallProposal(tool_name="search_repo", raw_arguments={"query": "quantity"}), trace_id="t-1"
        )
        assert result.success is True


class TestDangerousToolsAreApprovalGated:
    async def test_apply_patch_is_denied_without_approval(self, tmp_path):
        config = make_config(tmp_path)
        sandbox = make_sandbox(tmp_path)
        ctx = build_eng_context(config, model_catalog_path=REPO_ROOT_CATALOG, repo_dir=sandbox)
        result = await ctx.tool_executor.execute(
            ToolCallProposal(
                tool_name="apply_patch",
                raw_arguments={"patch_text": VALID_PATCH, "expected_file_path": "inventory/stock.py"},
            ),
            trace_id="t-1",
        )
        assert result.success is False
        assert "cannot remove more than current quantity" not in (sandbox / "inventory" / "stock.py").read_text()


class TestFullFixThenApproveThenApplyThenTestRoundTrip:
    async def test_the_real_bug_is_fixed_and_verified(self, tmp_path):
        config = make_config(tmp_path)
        sandbox = make_sandbox(tmp_path)
        runtime = FakeRuntime(default_response=VALID_PATCH)
        ctx = build_eng_context(
            config,
            model_catalog_path=REPO_ROOT_CATALOG,
            repo_dir=sandbox,
            runtime=runtime,
            approval_gate=AutoApprovalGate(),
        )

        # Before: the real, currently-failing test fails.
        before = await ctx.tool_executor.execute(
            ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="t-1"
        )
        assert before.data["passed"] is False

        # Propose, then apply, the real patch.
        proposed = await ctx.tool_executor.execute(
            ToolCallProposal(
                tool_name="propose_patch",
                raw_arguments={"instruction": "fix remove_stock", "file_contents": {"inventory/stock.py": "..."}},
            ),
            trace_id="t-2",
        )
        applied = await ctx.tool_executor.execute(
            ToolCallProposal(
                tool_name="apply_patch",
                raw_arguments={"patch_text": proposed.data, "expected_file_path": "inventory/stock.py"},
            ),
            trace_id="t-3",
        )
        assert applied.success is True

        # After: the full suite passes.
        after = await ctx.tool_executor.execute(
            ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="t-4"
        )
        assert after.data["passed"] is True

        # Every call was audit logged.
        entries = ctx.base.audit_log.all_entries()
        assert len(entries) == 4
        assert [e.tool_name for e in entries] == ["run_tests", "propose_patch", "apply_patch", "run_tests"]
