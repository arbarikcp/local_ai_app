from pydantic import BaseModel, Field

from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate
from local_ai_agents.policies.audit_log import AuditLog
from local_ai_agents.policies.budgets import ToolBudget
from local_ai_agents.policies.permissions import PermissionPolicy
from local_ai_agents.tools.base import Tool, ToolCallProposal
from local_ai_agents.tools.registry import ToolRegistry


class AddArgs(BaseModel):
    a: int
    b: int = Field(ge=0)


async def add_handler(args: AddArgs) -> int:
    return args.a + args.b


async def raising_handler(args: AddArgs) -> int:
    raise RuntimeError("handler exploded")


def make_registry(handler=add_handler, dangerous: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="add", description="add two numbers", args_model=AddArgs, handler=handler, dangerous=dangerous))
    return registry


class TestHappyPath:
    async def test_executes_and_returns_success(self):
        executor = ToolExecutor(make_registry())
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 2, "b": 3}))
        assert result.success is True
        assert result.data == 5


class TestUnknownTool:
    async def test_denies_a_call_to_an_unregistered_tool(self):
        executor = ToolExecutor(make_registry())
        result = await executor.execute(ToolCallProposal(tool_name="missing"))
        assert result.success is False
        assert "No tool registered" in result.error_message


class TestPermissions:
    async def test_denies_a_role_without_permission(self):
        permissions = PermissionPolicy()
        permissions.allow("analyst", "add")
        executor = ToolExecutor(make_registry(), permissions=permissions)
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}), role="guest")
        assert result.success is False
        assert "not permitted" in result.error_message

    async def test_allows_a_role_with_permission(self):
        permissions = PermissionPolicy()
        permissions.allow("analyst", "add")
        executor = ToolExecutor(make_registry(), permissions=permissions)
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}), role="analyst")
        assert result.success is True


class TestArgumentValidation:
    async def test_invalid_arguments_are_rejected_before_the_handler_runs(self):
        executor = ToolExecutor(make_registry())
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": -5}))
        assert result.success is False
        assert "argument validation failed" in result.error_message

    async def test_missing_required_arguments_are_rejected(self):
        executor = ToolExecutor(make_registry())
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1}))
        assert result.success is False


class TestDangerousToolsAndApproval:
    async def test_dangerous_tool_is_denied_by_default_null_approval_gate(self):
        executor = ToolExecutor(make_registry(dangerous=True))
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        assert result.success is False
        assert "not approved" in result.error_message

    async def test_dangerous_tool_succeeds_with_an_approving_gate(self):
        executor = ToolExecutor(make_registry(dangerous=True), approval_gate=AutoApprovalGate())
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        assert result.success is True

    async def test_non_dangerous_tool_never_consults_the_approval_gate(self):
        executor = ToolExecutor(make_registry(dangerous=False), approval_gate=NullApprovalGate())
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        assert result.success is True


class TestBudget:
    async def test_denies_once_the_budget_is_exhausted(self):
        budget = ToolBudget(max_total_calls=1)
        executor = ToolExecutor(make_registry(), budget=budget)
        first = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        second = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        assert first.success is True
        assert second.success is False
        assert "budget" in second.error_message.lower()


class TestHandlerErrors:
    async def test_a_handler_exception_is_wrapped_not_left_uncaught(self):
        executor = ToolExecutor(make_registry(handler=raising_handler))
        result = await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))
        assert result.success is False
        assert "handler exploded" in result.error_message


class TestAuditLogging:
    async def test_every_attempt_is_logged_including_denials(self, tmp_path):
        audit_log = AuditLog(tmp_path / "audit.db")
        permissions = PermissionPolicy()  # nothing allowed
        executor = ToolExecutor(make_registry(), permissions=permissions, audit_log=audit_log)

        await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}), trace_id="t1")

        entries = audit_log.entries_for_trace("t1")
        assert len(entries) == 1
        assert entries[0].outcome == "denied"
        audit_log.close()

    async def test_successful_calls_are_logged_too(self, tmp_path):
        audit_log = AuditLog(tmp_path / "audit.db")
        executor = ToolExecutor(make_registry(), audit_log=audit_log)

        await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}), trace_id="t1")

        entries = audit_log.entries_for_trace("t1")
        assert entries[0].outcome == "success"
        audit_log.close()

    async def test_a_generated_trace_id_is_used_when_none_is_provided(self, tmp_path):
        audit_log = AuditLog(tmp_path / "audit.db")
        executor = ToolExecutor(make_registry(), audit_log=audit_log)

        await executor.execute(ToolCallProposal(tool_name="add", raw_arguments={"a": 1, "b": 1}))

        assert len(audit_log.all_entries()) == 1
        audit_log.close()
