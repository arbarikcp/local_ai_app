"""Shared fixture setup for Module 15's lab scripts - the same "how many
open tickets are there?" task, solved two ways (Lab 1's ReAct loop and
Lab 3's workflow graph) so the difference between the two shapes is
measurable on identical data, not just asserted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from local_ai_core.runtimes.types import LLMRequest, LLMResponse
from local_ai_agents.tools.registry import ToolRegistry
from local_ai_agents.tools.sql_query import make_sql_query_tool


class ScriptedTurnRuntime:
    """Returns one scripted response per call, in order (repeating the
    last one if exhausted) - a real, deterministic multi-turn stand-in
    `FakeRuntime` can't provide on its own (it returns identical text on
    every call for a given model, regardless of turn).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        text = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return LLMResponse(
            text=text, model=request.model, prompt_tokens=10, completion_tokens=10, latency_ms=0.0, stop_reason="stop"
        )

    async def stream(self, request: LLMRequest):  # pragma: no cover - not used by these labs
        raise NotImplementedError("ScriptedTurnRuntime does not support streaming")

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        return [0] * len(rendered_prompt.split())


def make_fixture_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, subject TEXT, status TEXT)")
    conn.executemany(
        "INSERT INTO tickets (subject, status) VALUES (?, ?)",
        [
            ("Password reset not working", "open"),
            ("Billing question", "closed"),
            ("API rate limit hit", "open"),
            ("Cannot upload file", "open"),
            ("Refund request", "closed"),
        ],
    )
    conn.commit()
    conn.close()


def make_registry(db_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_sql_query_tool(db_path))
    return registry
