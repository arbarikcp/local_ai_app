"""CheckpointStore (theory doc §11) - real SQLite persistence (same
pattern as Module 8.5's SessionStore, Module 14's AuditLog): a workflow's
current node and state are saved after every step, and a new
`WorkflowExecutor` can resume a run from its last checkpoint after an
actual process restart, proven not asserted.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    node_name: str
    state: dict[str, Any]
    step_index: int
    timestamp: str


class CheckpointStore:
    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT PRIMARY KEY,
                node_name TEXT NOT NULL,
                state TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, run_id: str, node_name: str, state: dict[str, Any], step_index: int) -> None:
        self._conn.execute(
            "INSERT INTO checkpoints (run_id, node_name, state, step_index, timestamp) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET node_name=excluded.node_name, state=excluded.state, "
            "step_index=excluded.step_index, timestamp=excluded.timestamp",
            (run_id, node_name, json.dumps(state), step_index, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def load(self, run_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT run_id, node_name, state, step_index, timestamp FROM checkpoints WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(run_id=row[0], node_name=row[1], state=json.loads(row[2]), step_index=row[3], timestamp=row[4])

    def delete(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
