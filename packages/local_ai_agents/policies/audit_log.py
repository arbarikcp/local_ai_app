"""AuditLog (theory doc §11) - real SQLite persistence (stdlib `sqlite3`,
no server), same pattern as Module 8.5's `SessionStore`: every tool call
attempt (allowed or denied, succeeded or failed) is logged with a trace
id, arguments, outcome, and timestamp - proven across an actual
close/reopen cycle, not asserted from an in-memory list.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class AuditEntry:
    trace_id: str
    tool_name: str
    arguments: dict
    outcome: str
    detail: str
    timestamp: str


class AuditLog:
    def __init__(self, db_path: Path | str) -> None:
        # check_same_thread=False: a caller that builds this once (e.g. at
        # app-startup composition-root time) and then logs from inside a
        # request handler crosses threads under ASGI servers that dispatch
        # sync handlers to a worker pool, or under test clients that run
        # the app in a dedicated event-loop thread (Project 1's own
        # FastAPI test suite hit this for real: sqlite3.ProgrammingError,
        # "objects created in a thread can only be used in that same
        # thread"). Safe for the sequential-access pattern every caller in
        # this repo actually uses; true concurrent multi-threaded writes
        # would need an explicit lock or WAL mode.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(self, trace_id: str, tool_name: str, arguments: dict, outcome: str, detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO audit_entries (trace_id, tool_name, arguments, outcome, detail, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                tool_name,
                json.dumps(arguments),
                outcome,
                detail,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def _rows_to_entries(self, rows: list[tuple]) -> list[AuditEntry]:
        return [
            AuditEntry(trace_id=r[0], tool_name=r[1], arguments=json.loads(r[2]), outcome=r[3], detail=r[4], timestamp=r[5])
            for r in rows
        ]

    def entries_for_trace(self, trace_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT trace_id, tool_name, arguments, outcome, detail, timestamp FROM audit_entries "
            "WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        return self._rows_to_entries(rows)

    def all_entries(self) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT trace_id, tool_name, arguments, outcome, detail, timestamp FROM audit_entries ORDER BY id ASC"
        ).fetchall()
        return self._rows_to_entries(rows)

    def close(self) -> None:
        self._conn.close()
