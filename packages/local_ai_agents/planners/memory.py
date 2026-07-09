"""AgentMemory (theory doc §7) - the running step history within a single
workflow run, available to later decision points' prompts. Explicitly
scoped to *within one run*; cross-run/long-term memory is RAG-backed
memory (Module 11) or conversation memory (Module 8.5), not reinvented
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryEntry:
    step_index: int
    kind: str  # "reasoning" | "tool_call" | "observation" | "node_transition"
    content: str
    data: dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, kind: str, content: str, data: dict[str, Any] | None = None) -> MemoryEntry:
        entry = MemoryEntry(step_index=len(self._entries), kind=kind, content=content, data=data or {})
        self._entries.append(entry)
        return entry

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def transcript(self) -> str:
        """A plain-text rendering suitable for embedding in a prompt -
        every planner's decision points see the same run's history the
        same way.
        """
        return "\n".join(f"[{e.kind}] {e.content}" for e in self._entries)
