"""GwStorage — real, persistent SQLite per-request gateway log
(ARCHITECTURE.md "Gateway request log"). Same idiom as every prior
project's store: stdlib `sqlite3`, `check_same_thread=False` applied from
the start (Project 1's own real bug fix under FastAPI's `TestClient`
threading), frozen-dataclass records.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_requests (
    request_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    task TEXT NOT NULL,
    model_used TEXT NOT NULL,
    used_fallback INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

VALID_STATUSES = ("ok", "timeout", "queue_full", "no_runtimes_available")


@dataclass(frozen=True)
class GatewayRequestRecord:
    request_id: str
    trace_id: str
    task: str
    model_used: str
    used_fallback: bool
    latency_ms: float
    status: str
    created_at: str | None = None


class GwStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> GwStorage:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save_request(self, record: GatewayRequestRecord) -> None:
        if record.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {record.status!r}")
        self._conn.execute(
            "INSERT INTO gateway_requests (request_id, trace_id, task, model_used, used_fallback, "
            "latency_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.request_id,
                record.trace_id,
                record.task,
                record.model_used,
                int(record.used_fallback),
                record.latency_ms,
                record.status,
            ),
        )
        self._conn.commit()

    def get_request(self, request_id: str) -> GatewayRequestRecord | None:
        row = self._conn.execute(
            "SELECT request_id, trace_id, task, model_used, used_fallback, latency_ms, status, created_at "
            "FROM gateway_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_requests(self, *, task: str | None = None) -> list[GatewayRequestRecord]:
        if task is not None:
            rows = self._conn.execute(
                "SELECT request_id, trace_id, task, model_used, used_fallback, latency_ms, status, created_at "
                "FROM gateway_requests WHERE task = ? ORDER BY created_at DESC",
                (task,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT request_id, trace_id, task, model_used, used_fallback, latency_ms, status, created_at "
                "FROM gateway_requests ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> GatewayRequestRecord:
        return GatewayRequestRecord(
            request_id=row[0],
            trace_id=row[1],
            task=row[2],
            model_used=row[3],
            used_fallback=bool(row[4]),
            latency_ms=row[5],
            status=row[6],
            created_at=row[7],
        )
