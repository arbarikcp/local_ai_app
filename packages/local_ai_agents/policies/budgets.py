"""ToolBudget (theory doc §12) - per-session and per-tool call limits,
real counters that raise once exhausted. Never trust the model to stop
calling tools on its own - the same discipline Module 15's loop-prevention
topic builds on for full agent loops.
"""

from __future__ import annotations

from local_ai_agents.tools.base import ToolBudgetExceededError


class ToolBudget:
    def __init__(self, max_total_calls: int, max_calls_per_tool: int | None = None) -> None:
        self.max_total_calls = max_total_calls
        self.max_calls_per_tool = max_calls_per_tool
        self._total_calls = 0
        self._calls_by_tool: dict[str, int] = {}

    def consume(self, tool_name: str) -> None:
        if self._total_calls >= self.max_total_calls:
            raise ToolBudgetExceededError(f"Total tool-call budget exhausted ({self.max_total_calls} calls)")
        if self.max_calls_per_tool is not None and self._calls_by_tool.get(tool_name, 0) >= self.max_calls_per_tool:
            raise ToolBudgetExceededError(
                f"Per-tool budget exhausted for '{tool_name}' ({self.max_calls_per_tool} calls)"
            )
        self._total_calls += 1
        self._calls_by_tool[tool_name] = self._calls_by_tool.get(tool_name, 0) + 1

    @property
    def remaining_total_calls(self) -> int:
        return max(self.max_total_calls - self._total_calls, 0)

    def calls_for(self, tool_name: str) -> int:
        return self._calls_by_tool.get(tool_name, 0)
