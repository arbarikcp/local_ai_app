"""Labs 5-6 - human approval for a dangerous write tool, permissions, tool
budgets, and audit logs, all wired through `ToolExecutor`. Runs for real:
approval callbacks, permission checks, budget enforcement, and the audit
log's real SQLite persistence, no live model needed anywhere in this script.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_agents.executors.tool_executor import ToolExecutor  # noqa: E402
from local_ai_agents.policies.approval import CallbackApprovalGate, NullApprovalGate  # noqa: E402
from local_ai_agents.policies.audit_log import AuditLog  # noqa: E402
from local_ai_agents.policies.budgets import ToolBudget  # noqa: E402
from local_ai_agents.policies.permissions import PermissionPolicy  # noqa: E402
from local_ai_agents.tools.base import ToolCallProposal  # noqa: E402
from local_ai_agents.tools.registry import ToolRegistry  # noqa: E402
from local_ai_agents.tools.write_file import make_write_file_tool  # noqa: E402


async def deny_deletes_approve_notes(tool_name: str, arguments: dict) -> bool:
    """A stand-in for a real human approver: approves writes to files
    named 'notes*', denies anything else - a real (if scripted) decision
    function, not a hardcoded True/False.
    """
    path = arguments.get("path", "")
    return path.startswith("notes")


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as sandbox_dir, tempfile.TemporaryDirectory() as audit_dir:
        sandbox = Path(sandbox_dir)
        write_tool = make_write_file_tool(sandbox)

        registry = ToolRegistry()
        registry.register(write_tool)

        permissions = PermissionPolicy()
        permissions.allow("editor", "write_file")
        # "guest" role is never granted write_file - denied by permissions alone.

        audit_log = AuditLog(Path(audit_dir) / "audit.db")

        # Lab 5a: no approval gate configured -> fails closed (NullApprovalGate).
        no_gate_executor = ToolExecutor(
            registry, permissions=permissions, approval_gate=NullApprovalGate(), audit_log=audit_log
        )
        denied_by_default = await no_gate_executor.execute(
            ToolCallProposal(tool_name="write_file", raw_arguments={"path": "notes.txt", "content": "hello"}),
            role="editor",
            trace_id="no-gate",
        )

        # Lab 5b: a real (scripted) approval callback - approves "notes*", denies others.
        callback_executor = ToolExecutor(
            registry,
            permissions=permissions,
            approval_gate=CallbackApprovalGate(deny_deletes_approve_notes),
            audit_log=audit_log,
        )
        approved_write = await callback_executor.execute(
            ToolCallProposal(tool_name="write_file", raw_arguments={"path": "notes.txt", "content": "hello"}),
            role="editor",
            trace_id="approved",
        )
        denied_write = await callback_executor.execute(
            ToolCallProposal(tool_name="write_file", raw_arguments={"path": "secrets.txt", "content": "hello"}),
            role="editor",
            trace_id="denied-by-callback",
        )

        # Permission check: a role never granted write_file is denied before approval is even asked.
        permission_denied = await callback_executor.execute(
            ToolCallProposal(tool_name="write_file", raw_arguments={"path": "notes.txt", "content": "hello"}),
            role="guest",
            trace_id="permission-denied",
        )

        # Lab 6 (budget): a 2-call budget shared across two approved writes.
        budget = ToolBudget(max_total_calls=2)
        budget_executor = ToolExecutor(
            registry,
            permissions=permissions,
            approval_gate=CallbackApprovalGate(deny_deletes_approve_notes),
            budget=budget,
            audit_log=audit_log,
        )
        for i in range(3):
            await budget_executor.execute(
                ToolCallProposal(tool_name="write_file", raw_arguments={"path": f"notes_{i}.txt", "content": "x"}),
                role="editor",
                trace_id="budget-test",
            )

        return {
            "denied_by_default_null_gate": not denied_by_default.success,
            "approved_write_succeeded": approved_write.success,
            "denied_write_by_callback": not denied_write.success,
            "permission_denied_before_approval": not permission_denied.success,
            "budget_audit_outcomes": [e.outcome for e in audit_log.entries_for_trace("budget-test")],
            "sandbox_files_written": sorted(p.name for p in sandbox.glob("*.txt")),
            "total_audit_entries": len(audit_log.all_entries()),
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 5-6 - human approval, permissions, budgets, audit logs\n\n"
        f"- Denied by default (no real approval gate configured): {result['denied_by_default_null_gate']}\n"
        f"- Approved write ('notes.txt', callback approves): {result['approved_write_succeeded']}\n"
        f"- Denied write ('secrets.txt', callback denies): {result['denied_write_by_callback']}\n"
        f"- Denied by permissions before approval was even asked ('guest' role): {result['permission_denied_before_approval']}\n"
        f"- Budget test outcomes (3 attempts, 2-call budget): {result['budget_audit_outcomes']}\n"
        f"- Files actually written to the sandbox: {result['sandbox_files_written']}\n"
        f"- Total audit log entries recorded: {result['total_audit_entries']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
