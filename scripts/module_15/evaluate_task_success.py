"""Lab 6 - evaluate task success: a small golden set of (request, expected
final node, expected outcome key/value) cases run through the real
`WorkflowGraph` from `workflow_graph_demo.py`, scored by exact match on
the final state reached - Module 13's evaluation discipline (a golden set
+ a deterministic scorer) applied to agent workflows, not a vibe check.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from lab_fixtures import ScriptedTurnRuntime, make_fixture_database  # noqa: E402
from workflow_graph_demo import build_graph, make_budget  # noqa: E402

from local_ai_agents.executors.workflow_executor import WorkflowExecutor  # noqa: E402
from local_ai_agents.policies.approval import AutoApprovalGate  # noqa: E402


@dataclass(frozen=True)
class TaskGoldenCase:
    case_id: str
    scripted_summary: str
    expected_stopped_reason: str
    expected_open_ticket_count: int


GOLDEN_CASES = [
    TaskGoldenCase("normal_run", "There are 3 open tickets.", "end", 3),
    TaskGoldenCase("verbose_summary", "Currently there are 3 open support tickets needing attention.", "end", 3),
    # Deliberately wrong expectation (the fixture database has 3 open tickets,
    # not 99) - proves the scorer actually fails a case, not a rubber stamp.
    TaskGoldenCase("deliberately_wrong_expectation", "There are 3 open tickets.", "end", 99),
]


async def evaluate_case(case: TaskGoldenCase, db_path: Path, sandbox: Path) -> dict:
    runtime = ScriptedTurnRuntime([case.scripted_summary])
    graph = build_graph(db_path, sandbox, runtime)
    executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())
    result = await executor.run({}, make_budget())

    stopped_reason_correct = result.stopped_reason == case.expected_stopped_reason
    count_correct = result.final_state.get("open_ticket_count") == case.expected_open_ticket_count
    return {
        "case_id": case.case_id,
        "success": stopped_reason_correct and count_correct,
        "stopped_reason": result.stopped_reason,
        "open_ticket_count": result.final_state.get("open_ticket_count"),
    }


async def run_lab() -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        sandbox = Path(tmp_dir) / "sandbox"
        sandbox.mkdir()
        make_fixture_database(db_path)

        return [await evaluate_case(case, db_path, sandbox) for case in GOLDEN_CASES]


def result_to_markdown(rows: list[dict]) -> str:
    successes = sum(1 for r in rows if r["success"])
    lines = ["# Lab 6 - agent task success evaluation\n"]
    lines.append(f"- Task success rate: {successes}/{len(rows)}\n")
    for row in rows:
        lines.append(f"- {row['case_id']}: success={row['success']}, stopped_reason={row['stopped_reason']}, open_ticket_count={row['open_ticket_count']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    rows = asyncio.run(run_lab())
    print(result_to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
