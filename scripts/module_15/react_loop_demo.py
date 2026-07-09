"""Labs 1-2 - a real ReAct loop solving "how many open tickets are
there?" over a real SQLite fixture database, then a real, reproducible
break: an adversarial scripted runtime that always proposes the same tool
call, provoking `LoopGuard` to trip. Runs for real except the LLM turns
themselves (`ScriptedTurnRuntime`, no live model needed).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from lab_fixtures import ScriptedTurnRuntime, make_fixture_database, make_registry  # noqa: E402

from local_ai_agents.executors.tool_executor import ToolExecutor  # noqa: E402
from local_ai_agents.planners.loop_prevention import LoopGuard  # noqa: E402
from local_ai_agents.planners.react_loop import ReActLoop  # noqa: E402
from local_ai_agents.planners.safety_budget import AgentSafetyBudget  # noqa: E402

REQUEST = "How many open tickets are there?"

HAPPY_PATH_RESPONSES = [
    '{"action": "tool_call", "tool": "sql_query", "arguments": {"query": "SELECT COUNT(*) as n FROM tickets WHERE status = \'open\'"}}',
    '{"action": "final_answer", "answer": "There are 3 open tickets."}',
]

# An adversarial scenario: the model keeps re-issuing the exact same query
# instead of ever producing a final answer - a real, reproducible failure
# mode of the "avoid" shape, not a hypothetical one.
ADVERSARIAL_RESPONSES = [
    '{"action": "tool_call", "tool": "sql_query", "arguments": {"query": "SELECT COUNT(*) as n FROM tickets WHERE status = \'open\'"}}'
]


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        make_fixture_database(db_path)
        registry = make_registry(db_path)
        executor = ToolExecutor(registry)

        # Lab 1: a real, successful ReAct run.
        happy_runtime = ScriptedTurnRuntime(HAPPY_PATH_RESPONSES)
        happy_loop = ReActLoop(registry, executor, happy_runtime, model="fake-model")
        happy_budget = AgentSafetyBudget(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)
        happy_result = await happy_loop.run(REQUEST, happy_budget)

        # Lab 2: the same loop, an adversarial runtime that never stops proposing
        # the same tool call. LoopGuard (not the safety budget) is what actually
        # catches it here - max_repeats=3 trips well before max_steps=8 would.
        adversarial_runtime = ScriptedTurnRuntime(ADVERSARIAL_RESPONSES)
        adversarial_loop = ReActLoop(
            registry, executor, adversarial_runtime, model="fake-model", loop_guard=LoopGuard(max_repeats=3)
        )
        adversarial_budget = AgentSafetyBudget(
            max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000
        )
        adversarial_result = await adversarial_loop.run(REQUEST, adversarial_budget)

        return {
            "request": REQUEST,
            "happy_final_answer": happy_result.final_answer,
            "happy_stopped_reason": happy_result.stopped_reason,
            "happy_steps": len(happy_result.memory),
            "adversarial_stopped_reason": adversarial_result.stopped_reason,
            "adversarial_llm_calls_made": adversarial_runtime.call_count,
            "adversarial_would_have_looped_forever": adversarial_runtime.call_count < 8,
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 1-2 - ReAct loop, then broken by an adversarial prompt\n\n"
        f"- Request: {result['request']}\n"
        f"## Lab 1: happy path\n"
        f"- Stopped reason: {result['happy_stopped_reason']}\n"
        f"- Final answer: {result['happy_final_answer']}\n"
        f"- Memory entries: {result['happy_steps']}\n"
        f"## Lab 2: adversarial break\n"
        f"- Stopped reason: {result['adversarial_stopped_reason']}\n"
        f"- LLM calls made before LoopGuard tripped: {result['adversarial_llm_calls_made']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
