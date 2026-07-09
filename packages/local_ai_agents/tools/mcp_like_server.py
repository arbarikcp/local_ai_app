"""McpLikeServer (theory doc §5) - a real, in-process method dispatcher
shaped like MCP's tools/resources/prompts primitives. Every `tools/call`
request is routed through Module 14's `ToolExecutor`, never a tool
handler directly - the security boundary (theory doc §8) made
structurally true, not just claimed. Not a spec-compliant MCP
implementation (no JSON-RPC 2.0 transport, no capability negotiation) -
explicitly "MCP-*like*", matching this file's own name.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from local_ai_core.evals.prompt_injection import detect_prompt_injection_patterns
from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.policies.audit_log import AuditLog
from local_ai_agents.tools.base import ToolCallProposal
from local_ai_agents.tools.mcp_prompts import PromptRegistry
from local_ai_agents.tools.mcp_resources import ResourceRegistry
from local_ai_agents.tools.registry import ToolRegistry


class McpMethodNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class McpRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResponse:
    success: bool
    result: Any = None
    error_message: str | None = None


class McpLikeServer:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        *,
        resource_registry: ResourceRegistry | None = None,
        prompt_registry: PromptRegistry | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._resource_registry = resource_registry
        self._prompt_registry = prompt_registry
        self._audit_log = audit_log

    async def dispatch(self, request: McpRequest, *, role: str = "default") -> McpResponse:
        trace_id = str(uuid.uuid4())
        try:
            response = await self._dispatch_inner(request, role)
        except Exception as exc:  # noqa: BLE001 - every dispatch failure is logged, never silently lost
            self._log(trace_id, request, "error", str(exc))
            return McpResponse(success=False, error_message=str(exc))

        self._log(trace_id, request, "success" if response.success else "denied", response.error_message or "")
        return response

    async def _dispatch_inner(self, request: McpRequest, role: str) -> McpResponse:
        if request.method == "tools/list":
            schemas = [self._screen_tool_schema(s) for s in self._tool_registry.schema_list()]
            return McpResponse(success=True, result=schemas)

        if request.method == "tools/call":
            tool_name = request.params["tool"]
            arguments = request.params.get("arguments", {})
            result = await self._tool_executor.execute(
                ToolCallProposal(tool_name=tool_name, raw_arguments=arguments), role=role
            )
            return McpResponse(success=result.success, result=result.data, error_message=result.error_message)

        if request.method == "resources/list":
            if self._resource_registry is None:
                return McpResponse(success=True, result=[])
            return McpResponse(success=True, result=self._resource_registry.list())

        if request.method == "resources/read":
            if self._resource_registry is None:
                return McpResponse(success=False, error_message="No resources are registered on this server")
            content = self._resource_registry.read(request.params["uri"])
            flagged = detect_prompt_injection_patterns(content.text)
            return McpResponse(
                success=True, result={"uri": content.uri, "text": content.text, "flagged_patterns": flagged}
            )

        if request.method == "prompts/list":
            if self._prompt_registry is None:
                return McpResponse(success=True, result=[])
            return McpResponse(success=True, result=self._prompt_registry.list())

        if request.method == "prompts/get":
            if self._prompt_registry is None:
                return McpResponse(success=False, error_message="No prompts are registered on this server")
            rendered = self._prompt_registry.get(request.params["name"], request.params.get("arguments"))
            return McpResponse(success=True, result=rendered)

        raise McpMethodNotFoundError(f"Unknown MCP method '{request.method}'")

    def _screen_tool_schema(self, schema: dict) -> dict:
        """Tool descriptions are prompt surface area (theory doc's own
        gotcha) - screened the same way resource content is, before ever
        being exposed via `tools/list`.
        """
        flagged = detect_prompt_injection_patterns(schema.get("description", ""))
        return {**schema, "flagged_patterns": flagged}

    def _log(self, trace_id: str, request: McpRequest, outcome: str, detail: str) -> None:
        if self._audit_log is not None:
            self._audit_log.record(trace_id, request.method, request.params, outcome, detail)
