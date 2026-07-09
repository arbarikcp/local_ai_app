"""Labs 1-4 - build a tool registry with a calculator, a file search tool
(over the real Nimbus handbook corpus), and a read-only SQL tool (over a
real SQLite fixture database), then demonstrate LLM-proposed tool
selection and end-to-end deterministic execution. Runs for real except
`propose_tool_call()`'s generation, which uses `FakeRuntime`.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_agents.executors.tool_executor import ToolExecutor  # noqa: E402
from local_ai_agents.tools.base import ToolCallProposal  # noqa: E402
from local_ai_agents.tools.calculator import calculator_tool  # noqa: E402
from local_ai_agents.tools.file_search import make_file_search_tool  # noqa: E402
from local_ai_agents.tools.registry import ToolRegistry  # noqa: E402
from local_ai_agents.tools.sql_query import make_sql_query_tool  # noqa: E402
from local_ai_agents.tools.tool_call import propose_tool_call  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "datasets" / "rag_docs" / "nimbus_handbook"


def make_fixture_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, subject TEXT, status TEXT)")
    conn.executemany(
        "INSERT INTO tickets (subject, status) VALUES (?, ?)",
        [("Password reset not working", "open"), ("Billing question", "closed"), ("API rate limit hit", "open")],
    )
    conn.commit()
    conn.close()


def build_registry(db_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator_tool)
    registry.register(make_file_search_tool(CORPUS_DIR))
    registry.register(make_sql_query_tool(db_path))
    return registry


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        make_fixture_database(db_path)
        registry = build_registry(db_path)
        executor = ToolExecutor(registry)

        # Lab 1: registry contents.
        schema_names = [t.name for t in registry.list_tools()]

        # Lab 2: calculator, executed directly (no LLM needed to prove the tool works).
        calc_result = await executor.execute(ToolCallProposal(tool_name="calculator", raw_arguments={"expression": "(2 + 3) * 4"}))

        # Lab 3: file search, executed directly.
        search_result = await executor.execute(
            ToolCallProposal(tool_name="search_files", raw_arguments={"query": "15 minutes", "max_results": 5})
        )

        # Lab 4: SQL read-only tool, executed directly.
        sql_result = await executor.execute(
            ToolCallProposal(tool_name="sql_query", raw_arguments={"query": "SELECT subject FROM tickets WHERE status = 'open'"})
        )
        sql_write_attempt = await executor.execute(
            ToolCallProposal(tool_name="sql_query", raw_arguments={"query": "DELETE FROM tickets"})
        )

        # Lab 4 (tool selection): the LLM proposes which tool to call; the executor still enforces everything.
        runtime = FakeRuntime(default_response='{"tool": "calculator", "arguments": {"expression": "12 * 12"}}')
        proposal = await propose_tool_call("What is 12 times 12?", registry, runtime, model="fake-model")
        proposed_result = await executor.execute(proposal)

        return {
            "registered_tools": schema_names,
            "calculator_result": calc_result.as_text(),
            "file_search_result": search_result.data,
            "sql_select_result": sql_result.data,
            "sql_write_attempt_denied": not sql_write_attempt.success,
            "sql_write_attempt_error": sql_write_attempt.error_message,
            "llm_proposed_tool": proposal.tool_name if proposal else None,
            "llm_proposed_result": proposed_result.as_text(),
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 1-4 - tool registry, calculator, file search, SQL read-only tool\n\n"
        f"- Registered tools: {result['registered_tools']}\n"
        f"- calculator((2+3)*4): {result['calculator_result']}\n"
        f"- search_files('15 minutes'): {result['file_search_result']}\n"
        f"- sql_query(open tickets): {result['sql_select_result']}\n"
        f"- sql_query(DELETE ...) denied: {result['sql_write_attempt_denied']} — {result['sql_write_attempt_error']}\n"
        f"- LLM-proposed tool: {result['llm_proposed_tool']} -> {result['llm_proposed_result']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
