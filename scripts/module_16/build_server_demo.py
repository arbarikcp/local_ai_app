"""Labs 1-4 - build a tiny local MCP-like tool server exposing a file
search tool, a read-only SQL tool, a sandboxed resource, and a
parameterized prompt. Runs for real - no live model needed.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from server_fixtures import build_server, make_fixture_database  # noqa: E402

from local_ai_agents.tools.mcp_like_server import McpRequest  # noqa: E402


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        make_fixture_database(db_path)
        server = build_server(db_path)

        # Lab 1 + Lab 4: the server, and every tool's real metadata.
        tools_response = await server.dispatch(McpRequest(method="tools/list"))

        # Lab 2: file search tool, over the real Nimbus handbook corpus.
        search_response = await server.dispatch(
            McpRequest(method="tools/call", params={"tool": "search_files", "arguments": {"query": "15 minutes"}})
        )

        # Lab 3: read-only SQL tool, over a real SQLite fixture database.
        sql_response = await server.dispatch(
            McpRequest(
                method="tools/call",
                params={"tool": "sql_query", "arguments": {"query": "SELECT COUNT(*) as n FROM tickets WHERE status = 'open'"}},
            )
        )

        # Resources and prompts (theory doc §2-3).
        resources_response = await server.dispatch(McpRequest(method="resources/list"))
        resource_read_response = await server.dispatch(
            McpRequest(method="resources/read", params={"uri": "password_reset.md"})
        )
        prompts_response = await server.dispatch(McpRequest(method="prompts/list"))
        prompt_get_response = await server.dispatch(
            McpRequest(
                method="prompts/get",
                params={
                    "name": "rag_answer",
                    "arguments": {"context": "Reset links expire in 15 minutes.", "question": "How long is a reset link valid?"},
                },
            )
        )

        return {
            "tool_names": [t["name"] for t in tools_response.result],
            "tool_metadata_example": tools_response.result[0],
            "search_result": search_response.result,
            "sql_result": sql_response.result,
            "resource_uris": [r.uri for r in resources_response.result],
            "resource_content_length": len(resource_read_response.result["text"]),
            "prompt_names": [p.name for p in prompts_response.result],
            "rendered_prompt": prompt_get_response.result,
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 1-4 - a tiny local MCP-like server, real tools/resources/prompts\n\n"
        f"- Registered tools: {result['tool_names']}\n"
        f"- Example tool metadata: {result['tool_metadata_example']}\n"
        f"- search_files('15 minutes'): {result['search_result']}\n"
        f"- sql_query(open tickets): {result['sql_result']}\n"
        f"- Registered resources: {result['resource_uris']}\n"
        f"- Resource content length (real file read): {result['resource_content_length']} chars\n"
        f"- Registered prompts: {result['prompt_names']}\n"
        f"- Rendered prompt:\n{result['rendered_prompt']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
