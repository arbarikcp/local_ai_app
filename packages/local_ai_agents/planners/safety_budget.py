"""AgentSafetyBudget (theory doc "Agent safety budget") - curriculum's
exact YAML shape, enforced by real counters and real wall-clock time, not
trusted to the model's own judgment about when to stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class SafetyBudgetExceededError(Exception):
    pass


@dataclass
class AgentSafetyBudget:
    max_steps: int
    max_tool_calls: int
    max_runtime_seconds: float
    max_tokens_total: int
    requires_human_approval: list[str] = field(default_factory=list)

    _steps_taken: int = field(default=0, init=False)
    _tool_calls_made: int = field(default=0, init=False)
    _tokens_used: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)

    def record_step(self) -> None:
        self._steps_taken += 1
        if self._steps_taken > self.max_steps:
            raise SafetyBudgetExceededError(f"max_steps exceeded ({self.max_steps})")

    def record_tool_call(self) -> None:
        self._tool_calls_made += 1
        if self._tool_calls_made > self.max_tool_calls:
            raise SafetyBudgetExceededError(f"max_tool_calls exceeded ({self.max_tool_calls})")

    def record_tokens(self, count: int) -> None:
        self._tokens_used += count
        if self._tokens_used > self.max_tokens_total:
            raise SafetyBudgetExceededError(f"max_tokens_total exceeded ({self.max_tokens_total})")

    def check_runtime(self) -> None:
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.max_runtime_seconds:
            raise SafetyBudgetExceededError(f"max_runtime_seconds exceeded ({self.max_runtime_seconds})")

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self.requires_human_approval

    @property
    def steps_taken(self) -> int:
        return self._steps_taken

    @property
    def tool_calls_made(self) -> int:
        return self._tool_calls_made

    @property
    def tokens_used(self) -> int:
        return self._tokens_used
