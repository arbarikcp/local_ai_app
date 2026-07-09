"""ToolExecutor - the deterministic code the theory doc's tool execution
rule describes: for every proposed tool call, is this tool allowed? Are
arguments valid? Is the user authorized? Is approval required? Every
attempt - successful, denied, or errored - is logged to the audit log if
one is configured, not just the ones that succeed.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError

from local_ai_agents.policies.approval import ApprovalGate, NullApprovalGate
from local_ai_agents.policies.audit_log import AuditLog
from local_ai_agents.policies.budgets import ToolBudget
from local_ai_agents.policies.permissions import PermissionPolicy
from local_ai_agents.tools.base import ToolBudgetExceededError, ToolCallProposal, ToolNotFoundError, ToolResult
from local_ai_agents.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permissions: PermissionPolicy | None = None,
        approval_gate: ApprovalGate | None = None,
        budget: ToolBudget | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._registry = registry
        self._permissions = permissions
        self._approval_gate: ApprovalGate = approval_gate or NullApprovalGate()
        self._budget = budget
        self._audit_log = audit_log

    async def execute(
        self, proposal: ToolCallProposal, *, role: str = "default", trace_id: str | None = None
    ) -> ToolResult:
        trace_id = trace_id or str(uuid.uuid4())
        arguments = proposal.raw_arguments

        try:
            tool = self._registry.get(proposal.tool_name)
        except ToolNotFoundError as exc:
            return self._deny(trace_id, proposal.tool_name, arguments, str(exc))

        if self._permissions is not None and not self._permissions.is_allowed(role, tool.name):
            return self._deny(
                trace_id, tool.name, arguments, f"role '{role}' is not permitted to call '{tool.name}'"
            )

        try:
            validated_args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return self._deny(trace_id, tool.name, arguments, f"argument validation failed: {exc}", outcome="error")

        if tool.dangerous:
            approved = await self._approval_gate.request_approval(tool.name, arguments)
            if not approved:
                return self._deny(trace_id, tool.name, arguments, "dangerous tool call was not approved")

        if self._budget is not None:
            try:
                self._budget.consume(tool.name)
            except ToolBudgetExceededError as exc:
                return self._deny(trace_id, tool.name, arguments, str(exc))

        try:
            data = await tool.handler(validated_args)
        except Exception as exc:  # noqa: BLE001 - deliberately wrapped, never left uncaught
            self._log(trace_id, tool.name, arguments, "error", str(exc))
            return ToolResult(success=False, error_message=f"Tool execution failed: {exc}")

        self._log(trace_id, tool.name, arguments, "success", "")
        return ToolResult(success=True, data=data)

    def _deny(self, trace_id: str, tool_name: str, arguments: dict[str, Any], detail: str, outcome: str = "denied") -> ToolResult:
        self._log(trace_id, tool_name, arguments, outcome, detail)
        return ToolResult(success=False, error_message=detail)

    def _log(self, trace_id: str, tool_name: str, arguments: dict[str, Any], outcome: str, detail: str) -> None:
        if self._audit_log is not None:
            self._audit_log.record(trace_id, tool_name, arguments, outcome, detail)
