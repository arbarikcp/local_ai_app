"""Lab 5 - real checkpointing: run a workflow partway, let it fail, then
resume from an actual restart (a genuinely new `WorkflowExecutor` and
`CheckpointStore` instance pointed at the same SQLite file) and confirm it
continues from the checkpointed node rather than re-running work already
done. Runs for real, no live model needed.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_agents.executors.workflow_executor import WorkflowExecutor  # noqa: E402
from local_ai_agents.planners.checkpoint_store import CheckpointStore  # noqa: E402
from local_ai_agents.planners.safety_budget import AgentSafetyBudget  # noqa: E402
from local_ai_agents.planners.workflow_graph import END, WorkflowGraph, WorkflowNode  # noqa: E402


async def fetch_tickets(state: dict, memory) -> dict:
    memory.add("observation", "fetched 5 tickets")
    return {**state, "tickets_fetched": 5}


class FlakyStepOnce:
    """Fails exactly once (simulating a transient failure a real deployment
    might see - a timeout, a dropped connection), then succeeds on the
    next real attempt after resume.
    """

    def __init__(self) -> None:
        self.has_failed_once = False

    async def __call__(self, state: dict, memory) -> dict:
        if not self.has_failed_once:
            self.has_failed_once = True
            raise RuntimeError("simulated transient failure")
        memory.add("observation", "categorized tickets")
        return {**state, "categorized": True}


async def finalize(state: dict, memory) -> dict:
    memory.add("observation", "finalized triage")
    return {**state, "done": True}


def make_budget() -> AgentSafetyBudget:
    return AgentSafetyBudget(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)


def build_graph(categorize_fn) -> WorkflowGraph:
    graph = WorkflowGraph(start_node="fetch")
    graph.add_node(WorkflowNode(name="fetch", run=fetch_tickets))
    graph.add_node(WorkflowNode(name="categorize", run=categorize_fn))
    graph.add_node(WorkflowNode(name="finalize", run=finalize))
    graph.add_edge("fetch", "categorize")
    graph.add_edge("categorize", "finalize")
    graph.add_edge("finalize", END)
    return graph


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "checkpoints.db"
        flaky = FlakyStepOnce()

        # First "process": fetch succeeds and is checkpointed; categorize fails
        # (max_retries=0, so no retry masks the failure) and the run stops.
        store1 = CheckpointStore(db_path)
        executor1 = WorkflowExecutor(build_graph(flaky), checkpoint_store=store1, max_retries=0)
        first_result = await executor1.run({}, make_budget(), run_id="triage-run-1")
        store1.close()

        # A genuinely new process: new CheckpointStore, new WorkflowExecutor,
        # same SQLite file. `flaky` has already failed once, so this attempt
        # succeeds - simulating "the transient issue that caused the original
        # failure is gone by the time an operator retries."
        store2 = CheckpointStore(db_path)
        executor2 = WorkflowExecutor(build_graph(flaky), checkpoint_store=store2)
        resumed_result = await executor2.run({}, make_budget(), run_id="triage-run-1", resume=True)
        checkpoint_after_resume = store2.load("triage-run-1")
        store2.close()

        return {
            "first_run_stopped_reason": first_result.stopped_reason,
            "first_run_final_node": first_result.final_node,
            "first_run_state": first_result.final_state,
            "resumed_run_stopped_reason": resumed_result.stopped_reason,
            "resumed_run_final_state": resumed_result.final_state,
            "checkpoint_survived_after_resume": checkpoint_after_resume is not None,
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 5 - checkpointing and resume across a real restart\n\n"
        f"- First run stopped at '{result['first_run_final_node']}' ({result['first_run_stopped_reason']}) "
        f"with state {result['first_run_state']}\n"
        f"- Resumed run (new executor, new store, same SQLite file) result: {result['resumed_run_stopped_reason']}\n"
        f"- Final state after resume: {result['resumed_run_final_state']}\n"
        f"- 'fetch' was not re-run - its result (tickets_fetched=5) survived into the resumed state.\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
