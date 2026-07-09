"""ToolRegistry (theory doc §3) - register, look up, and list tools. A
call for an unregistered name fails here, before validation, permissions,
approval, or budget checks ever run.
"""

from __future__ import annotations

from local_ai_agents.tools.base import Tool, ToolNotFoundError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"No tool registered with name '{name}'") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schema_list(self) -> list[dict]:
        return [tool.json_schema() for tool in self._tools.values()]
