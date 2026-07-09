"""LoopGuard (theory doc §9) - a real circuit breaker: if the same tool
name and arguments are proposed `max_repeats` times in a row, `LoopGuard`
trips and raises `LoopDetectedError`, regardless of what the model wants
to do next. This is what actually breaks Lab 2's adversarial prompt, not
a documented risk.
"""

from __future__ import annotations

import json
from typing import Any


class LoopDetectedError(Exception):
    pass


def _signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


class LoopGuard:
    def __init__(self, max_repeats: int = 3) -> None:
        if max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        self.max_repeats = max_repeats
        self._last_signature: str | None = None
        self._consecutive_count = 0

    def record(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Raises `LoopDetectedError` once the *same* (tool, arguments)
        pair has been proposed `max_repeats` times in a row - a different
        proposal in between resets the count, since the loop is only a
        problem when the model is stuck, not when it's making progress.
        """
        signature = _signature(tool_name, arguments)
        if signature == self._last_signature:
            self._consecutive_count += 1
        else:
            self._last_signature = signature
            self._consecutive_count = 1

        if self._consecutive_count >= self.max_repeats:
            raise LoopDetectedError(
                f"Tool '{tool_name}' proposed with identical arguments {self._consecutive_count} times in a row"
            )

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count
