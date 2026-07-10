"""EngAppContext — the composition root for this project, extending (not
replacing) Module 23's `AppContext` (ARCHITECTURE.md "Composition root").
Same extension pattern Projects 1 and 2 established. Every capability this
project exposes is registered into one `ToolRegistry` and called only
through `ToolExecutor`, so every call - not just `apply`/`run_tests`, the
only two Module 17's `WorkflowExecutor` gated - is audit-logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.policies.approval import ApprovalGate, NullApprovalGate
from local_ai_agents.policies.permissions import PermissionPolicy
from local_ai_agents.tools.list_symbols import make_list_symbols_tool
from local_ai_agents.tools.read_file import make_read_file_tool
from local_ai_agents.tools.registry import ToolRegistry
from local_ai_agents.tools.search_repo import make_search_repo_tool
from local_ai_core.deployment.app_context import AppContext, build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.base import LLMRuntime

from eng_tools import make_apply_patch_tool, make_propose_patch_tool, make_run_tests_tool

DEFAULT_ROLE = "default"


@dataclass
class EngAppContext:
    base: AppContext
    repo_dir: Path
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor


def build_eng_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    repo_dir: Path,
    runtime: LLMRuntime | None = None,
    approval_gate: ApprovalGate | None = None,
) -> EngAppContext:
    """`approval_gate` defaults to `NullApprovalGate` - fail-closed, Module
    14's own safe default - so a caller must explicitly opt into
    `AutoApprovalGate`/`CallbackApprovalGate` before `apply_patch`/
    `run_tests` can actually run.
    """
    base = build_app_context(config, model_catalog_path=model_catalog_path, runtime=runtime)

    registry = ToolRegistry()
    registry.register(make_list_symbols_tool(repo_dir))
    registry.register(make_search_repo_tool(repo_dir))
    registry.register(make_read_file_tool(repo_dir))
    registry.register(make_propose_patch_tool(base.runtime, base.config.models.default_code))
    registry.register(make_apply_patch_tool(repo_dir))
    registry.register(make_run_tests_tool(repo_dir))

    permissions = PermissionPolicy()
    permissions.allow_all(DEFAULT_ROLE)

    executor = ToolExecutor(
        registry,
        permissions=permissions,
        approval_gate=approval_gate or NullApprovalGate(),
        audit_log=base.audit_log,
    )

    return EngAppContext(base=base, repo_dir=repo_dir, tool_registry=registry, tool_executor=executor)
