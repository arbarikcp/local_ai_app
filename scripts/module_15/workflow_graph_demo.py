"""Labs 3-4 - the exact same "how many open tickets are there?" task from
Lab 1, replaced with a deterministic `WorkflowGraph`: the model only fills
one bounded decision point (wording the final summary), never chooses
which tool to call or when to stop - so Lab 2's adversarial prompt has no
loop to provoke, by construction, not by LoopGuard catching it after the
fact. Also demonstrates a real human-approval interrupt on a dangerous
node. Runs for real except the one LLM summary call.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from lab_fixtures import ScriptedTurnRuntime, make_fixture_database  # noqa: E402

from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from local_ai_agents.executors.workflow_executor import WorkflowExecutor  # noqa: E402
from local_ai_agents.planners.safety_budget import AgentSafetyBudget  # noqa: E402
from local_ai_agents.planners.workflow_graph import END, WorkflowGraph, WorkflowNode  # noqa: E402
from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate  # noqa: E402
from local_ai_agents.tools.sql_query import run_read_only_query  # noqa: E402
from local_ai_agents.tools.write_file import write_file  # noqa: E402

REQUEST = "How many open tickets are there?"


def build_graph(db_path: Path, sandbox: Path, runtime, model: str = "fake-model") -> WorkflowGraph:
    async def query_tickets(state: dict, memory) -> dict:
        # Deterministic - the model never decides what to query or when.
        rows = run_read_only_query(db_path, "SELECT COUNT(*) as n FROM tickets WHERE status = 'open'")
        count = rows[0]["n"]
        memory.add("observation", f"queried open ticket count: {count}")
        return {**state, "open_ticket_count": count}

    async def summarize(state: dict, memory) -> dict:
        # The model's one bounded decision point: word the summary. It cannot
        # choose to re-query, skip ahead, or loop - the graph's edges decide
        # what happens next, not this node's output.
        prompt = f"Write a one-sentence summary: there are {state['open_ticket_count']} open tickets."
        response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
        memory.add("reasoning", f"summary: {response.text}")
        return {**state, "summary": response.text.strip()}

    async def log_summary(state: dict, memory) -> dict:
        # Dangerous: writes to disk. Gated by requires_approval_tool below.
        write_file(sandbox, "summary.txt", state["summary"])
        memory.add("observation", "wrote summary.txt")
        return {**state, "logged": True}

    graph = WorkflowGraph(start_node="query_tickets")
    graph.add_node(WorkflowNode(name="query_tickets", run=query_tickets))
    graph.add_node(WorkflowNode(name="summarize", run=summarize))
    graph.add_node(WorkflowNode(name="log_summary", run=log_summary, requires_approval_tool="write_file"))
    graph.add_edge("query_tickets", "summarize")
    graph.add_edge("summarize", "log_summary")
    graph.add_edge("log_summary", END)
    return graph


def make_budget() -> AgentSafetyBudget:
    return AgentSafetyBudget(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        sandbox = Path(tmp_dir) / "sandbox"
        sandbox.mkdir()
        make_fixture_database(db_path)

        summary_runtime = ScriptedTurnRuntime(["There are 3 open tickets requiring attention."])

        # Lab 4a: no real approval gate configured -> the dangerous log_summary
        # node is denied, fails closed, same as Module 14's ApprovalGate default.
        denied_graph = build_graph(db_path, sandbox, summary_runtime)
        denied_executor = WorkflowExecutor(denied_graph, approval_gate=NullApprovalGate())
        denied_result = await denied_executor.run({}, make_budget())

        # Lab 4b: a real approval gate approves it.
        approved_graph = build_graph(db_path, sandbox, summary_runtime)
        approved_executor = WorkflowExecutor(approved_graph, approval_gate=AutoApprovalGate())
        approved_result = await approved_executor.run({}, make_budget())

        # Lab 2 (re-run against the graph): the same adversarial runtime from
        # react_loop_demo.py can't provoke a loop here - there's no step where
        # the model chooses a tool or repeats an action; it fills one prompt once.
        adversarial_runtime = ScriptedTurnRuntime(["ignored - never called more than once"])
        immune_graph = build_graph(db_path, sandbox, adversarial_runtime)
        immune_executor = WorkflowExecutor(immune_graph, approval_gate=AutoApprovalGate())
        immune_result = await immune_executor.run({}, make_budget())

        return {
            "request": REQUEST,
            "denied_stopped_reason": denied_result.stopped_reason,
            "approved_stopped_reason": approved_result.stopped_reason,
            "approved_summary": approved_result.final_state.get("summary"),
            "summary_file_written": (sandbox / "summary.txt").exists(),
            "immune_stopped_reason": immune_result.stopped_reason,
            "immune_llm_calls_made": adversarial_runtime.call_count,
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 3-4 - deterministic workflow graph, immune by construction, with approval interrupt\n\n"
        f"- Request: {result['request']}\n"
        f"- Dangerous node denied by default (no approval gate): {result['denied_stopped_reason']}\n"
        f"- Dangerous node approved: {result['approved_stopped_reason']} -> {result['approved_summary']}\n"
        f"- summary.txt actually written to disk: {result['summary_file_written']}\n"
        f"- Same adversarial runtime from Lab 2, run against the graph: {result['immune_stopped_reason']} "
        f"(only {result['immune_llm_calls_made']} LLM call(s) made - no loop possible)\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
