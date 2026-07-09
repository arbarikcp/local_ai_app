"""SessionStore — SQLite-backed conversation persistence (theory doc §8-9,
12). No server, no external dependency - Python's stdlib sqlite3 module.

Schema deliberately has no columns for retrieved-document content or
arbitrary application state (theory doc §10: "retrieved context is not
conversation memory, tool state is not conversation memory") - a caller
that wants to store those needs a separate store; this schema doesn't
quietly accommodate scope creep.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from .turn import Turn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    turn_group_id TEXT,
    sticky INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_index);
"""


class SessionStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create_session(self, session_id: str | None = None) -> str:
        """A session "exists" once it has turns; this just generates an id
        (or accepts a caller-chosen one) - there's no separate sessions
        table row to insert.
        """
        return session_id or str(uuid.uuid4())

    def append_turn(self, session_id: str, turn: Turn) -> None:
        next_index = self._next_turn_index(session_id)
        self._conn.execute(
            "INSERT INTO turns (session_id, turn_index, role, content, tool_call_id, turn_group_id, sticky) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                next_index,
                turn.role,
                turn.content,
                turn.tool_call_id,
                turn.turn_group_id,
                int(turn.sticky),
            ),
        )
        self._conn.commit()

    def _next_turn_index(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(turn_index) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()
        return (row[0] + 1) if row and row[0] is not None else 0

    def get_turns(self, session_id: str) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT role, content, tool_call_id, turn_group_id, sticky FROM turns "
            "WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [
            Turn(role=row[0], content=row[1], tool_call_id=row[2], turn_group_id=row[3], sticky=bool(row[4]))
            for row in rows
        ]

    def list_sessions(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT session_id FROM turns ORDER BY session_id").fetchall()
        return [row[0] for row in rows]

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM turns WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
        return row is not None

    def delete_session(self, session_id: str) -> None:
        """Lab 6's memory deletion command - removes every turn for a
        session (theory doc §12: memory privacy and deletion).
        """
        self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        self._conn.commit()
