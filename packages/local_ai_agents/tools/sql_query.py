"""Read-only SQL query tool (Lab 4) - defense in depth: the query text
must be a single `SELECT` statement with no forbidden keywords (checked
before execution), *and* the SQLite connection itself is opened read-only
via the `mode=ro` URI (enforced by SQLite itself, not just this tool's own
logic) - neither layer alone has to be perfect.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool


class SqlQueryArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    max_rows: int = Field(default=50, ge=1, le=500)


class UnsafeQueryError(Exception):
    pass


_SELECT_ONLY_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|TRIGGER)\b",
    re.IGNORECASE,
)


def validate_read_only_query(query: str) -> None:
    stripped = query.strip()
    if ";" in stripped.rstrip(";"):
        raise UnsafeQueryError("Multiple statements are not allowed.")
    if not _SELECT_ONLY_RE.match(stripped):
        raise UnsafeQueryError("Only SELECT statements are allowed.")
    if _FORBIDDEN_KEYWORDS_RE.search(stripped):
        raise UnsafeQueryError("Query contains a forbidden keyword.")


def run_read_only_query(database_path: Path, query: str, max_rows: int = 50) -> list[dict]:
    validate_read_only_query(query)
    uri = f"file:{database_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query)
        rows = cursor.fetchmany(max_rows)
        return [dict(row) for row in rows]
    finally:
        conn.close()


def make_sql_query_tool(database_path: Path) -> Tool:
    async def handler(args: SqlQueryArgs) -> list[dict]:
        return run_read_only_query(database_path, args.query, args.max_rows)

    return Tool(
        name="sql_query",
        description="Run a read-only SELECT query against the support database.",
        args_model=SqlQueryArgs,
        handler=handler,
    )
