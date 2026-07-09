"""Labs 3-5 - attack tool calling with an injected tool request, add
policy enforcement, add approval workflow. Reuses Module 14's real
`ToolExecutor` pipeline unchanged: an injected "delete all orders" call
proposed by a (simulated) compromised model is denied by real permission
policy before it ever reaches the dangerous handler, and a legitimate
dangerous call is gated by a real `ApprovalGate` instead of running
unconditionally.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from pydantic import BaseModel  # noqa: E402

from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate  # noqa: E402
from local_ai_agents.policies.audit_log import AuditLog  # noqa: E402
from local_ai_agents.policies.permissions import PermissionPolicy  # noqa: E402
from local_ai_agents.tools.base import Tool, ToolCallProposal  # noqa: E402
from local_ai_agents.tools.registry import ToolRegistry  # noqa: E402
from local_ai_agents.executors.tool_executor import ToolExecutor  # noqa: E402


class LookupOrderArgs(BaseModel):
    order_id: str


class DeleteAllOrdersArgs(BaseModel):
    confirm: bool = False


async def lookup_order_handler(args: LookupOrderArgs) -> dict:
    return {"order_id": args.order_id, "status": "shipped"}


async def delete_all_orders_handler(args: DeleteAllOrdersArgs) -> dict:
    return {"deleted": True}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup_order",
            description="Look up an order by ID.",
            args_model=LookupOrderArgs,
            handler=lookup_order_handler,
            dangerous=False,
        )
    )
    registry.register(
        Tool(
            name="delete_all_orders",
            description="Delete every order in the system. Irreversible.",
            args_model=DeleteAllOrdersArgs,
            handler=delete_all_orders_handler,
            dangerous=True,
        )
    )
    return registry


async def run_lab() -> dict:
    permissions = PermissionPolicy()
    permissions.allow("support_agent", "lookup_order")
    # deliberately NOT granted delete_all_orders - support_agent should
    # never reach it, injected or not.

    audit_log = AuditLog(":memory:")

    # Lab 3: an injected tool request - as if a malicious tool-output or
    # document instructed the model to call the dangerous tool directly.
    injected_registry = build_registry()
    injected_executor = ToolExecutor(
        injected_registry, permissions=permissions, approval_gate=NullApprovalGate(), audit_log=audit_log
    )
    injected_proposal = ToolCallProposal(tool_name="delete_all_orders", raw_arguments={"confirm": True})
    injected_result = await injected_executor.execute(injected_proposal, role="support_agent", trace_id="attack-1")

    # Lab 4/5: a legitimate dangerous call from an authorized role, gated
    # by a real approval workflow instead of running unconditionally.
    admin_permissions = PermissionPolicy()
    admin_permissions.allow_all("admin")
    approved_executor = ToolExecutor(
        build_registry(), permissions=admin_permissions, approval_gate=AutoApprovalGate(), audit_log=audit_log
    )
    legitimate_proposal = ToolCallProposal(tool_name="delete_all_orders", raw_arguments={"confirm": True})
    legitimate_result = await approved_executor.execute(legitimate_proposal, role="admin", trace_id="legit-1")

    audit_entries = audit_log.all_entries()

    return {
        "injected_call_succeeded": injected_result.success,
        "injected_call_error": injected_result.error_message,
        "legitimate_call_succeeded": legitimate_result.success,
        "audit_entry_count": len(audit_entries),
        "audit_outcomes": [e.outcome for e in audit_entries],
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 3-5 - injected tool request, policy enforcement, approval workflow\n\n"
        f"- Injected `delete_all_orders` call from `support_agent` role: "
        f"succeeded={result['injected_call_succeeded']} ({result['injected_call_error']})\n"
        f"- Legitimate `delete_all_orders` call from `admin` role (approval-gated): "
        f"succeeded={result['legitimate_call_succeeded']}\n"
        f"- Audit log recorded {result['audit_entry_count']} entries: {result['audit_outcomes']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
