"""Shared fixture setup for Module 16's lab scripts - a real support
ticket SQLite database (same shape as Module 15's) and the real Nimbus
handbook corpus, wired into a real `McpLikeServer`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.policies.audit_log import AuditLog
from local_ai_agents.tools.file_search import make_file_search_tool
from local_ai_agents.tools.mcp_like_server import McpLikeServer
from local_ai_agents.tools.mcp_prompts import PromptRegistry
from local_ai_agents.tools.mcp_resources import ResourceRegistry
from local_ai_agents.tools.registry import ToolRegistry
from local_ai_agents.tools.sql_query import make_sql_query_tool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "datasets" / "rag_docs" / "nimbus_handbook"

RAG_PROMPT_TEMPLATE = (
    "You are a question answering assistant.\n"
    "Answer only using the provided context.\n"
    'If the answer is not present in the context, say: "I don\'t know based on the provided documents."\n\n'
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)


def make_fixture_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, subject TEXT, status TEXT)")
    conn.executemany(
        "INSERT INTO tickets (subject, status) VALUES (?, ?)",
        [
            ("Password reset not working", "open"),
            ("Billing question", "closed"),
            ("API rate limit hit", "open"),
        ],
    )
    conn.commit()
    conn.close()


def build_server(db_path: Path, *, audit_log: AuditLog | None = None) -> McpLikeServer:
    registry = ToolRegistry()
    registry.register(make_file_search_tool(CORPUS_DIR))
    registry.register(make_sql_query_tool(db_path))

    resources = ResourceRegistry(CORPUS_DIR)
    resources.register("password_reset.md", "Nimbus's password reset support article")

    prompts = PromptRegistry()
    prompts.register(
        "rag_answer",
        RAG_PROMPT_TEMPLATE,
        "Module 11's minimal RAG prompt, exposed as a reusable MCP-shaped prompt",
        argument_names=["context", "question"],
    )

    executor = ToolExecutor(registry)
    return McpLikeServer(registry, executor, resource_registry=resources, prompt_registry=prompts, audit_log=audit_log)
