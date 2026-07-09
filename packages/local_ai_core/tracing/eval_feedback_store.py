"""EvalFeedbackStore — SQLite-backed evaluation logs and user feedback
(theory doc §11-12). Same pattern as Module 8.5's `SessionStore` and
Module 19's `AdapterRegistry`: no server, no external dependency, Python's
stdlib sqlite3 module, two append-only tables tied together by `trace_id`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

VALID_RATINGS = ("up", "down")


@dataclass(frozen=True)
class EvalRunRecord:
    trace_id: str
    metric_name: str
    score: float
    created_at: str | None = None


@dataclass(frozen=True)
class UserFeedbackRecord:
    trace_id: str
    rating: str
    comment: str | None = None
    created_at: str | None = None


class EvalFeedbackStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EvalFeedbackStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def log_eval_run(self, record: EvalRunRecord) -> None:
        self._conn.execute(
            "INSERT INTO eval_runs (trace_id, metric_name, score) VALUES (?, ?, ?)",
            (record.trace_id, record.metric_name, record.score),
        )
        self._conn.commit()

    def log_user_feedback(self, record: UserFeedbackRecord) -> None:
        if record.rating not in VALID_RATINGS:
            raise ValueError(f"rating must be one of {VALID_RATINGS}, got {record.rating!r}")
        self._conn.execute(
            "INSERT INTO user_feedback (trace_id, rating, comment) VALUES (?, ?, ?)",
            (record.trace_id, record.rating, record.comment),
        )
        self._conn.commit()

    def get_eval_runs_for_trace(self, trace_id: str) -> list[EvalRunRecord]:
        rows = self._conn.execute(
            "SELECT trace_id, metric_name, score, created_at FROM eval_runs WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        return [EvalRunRecord(trace_id=r[0], metric_name=r[1], score=r[2], created_at=r[3]) for r in rows]

    def get_feedback_for_trace(self, trace_id: str) -> list[UserFeedbackRecord]:
        rows = self._conn.execute(
            "SELECT trace_id, rating, comment, created_at FROM user_feedback WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        return [UserFeedbackRecord(trace_id=r[0], rating=r[1], comment=r[2], created_at=r[3]) for r in rows]

    def feedback_summary(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT rating, COUNT(*) FROM user_feedback GROUP BY rating").fetchall()
        return {rating: count for rating, count in rows}
