"""ExtractionStore — real SQLite persistence for every extraction request
(ARCHITECTURE.md "Storage schema"). Same idiom as Module 19's
`AdapterRegistry` and Module 21's `EvalFeedbackStore`: stdlib `sqlite3`, no
server, a frozen-dataclass record shape. The one query pattern that didn't
exist anywhere in the repo before this project: "list extractions that need
review," keyed on the pipeline's own `needs_review` flag rather than a raw
string match on confidence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    request_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    raw_input TEXT NOT NULL,
    extracted_fields TEXT NOT NULL,
    confidence TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    validation_error TEXT,
    used_repair_retry INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_extractions_needs_review ON extractions (needs_review);
"""


@dataclass(frozen=True)
class ExtractionRecord:
    request_id: str
    trace_id: str
    schema_name: str
    raw_input: str
    extracted_fields: dict[str, Any]
    confidence: str
    needs_review: bool
    validation_error: str | None
    used_repair_retry: bool
    latency_ms: float
    created_at: str | None = None


class ExtractionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        # check_same_thread=False: FastAPI dispatches sync (`def`, not
        # `async def`) endpoints to a worker-thread pool, so a connection
        # created once at context-build time (main thread) must remain
        # usable from whichever thread later handles a request - real bug
        # this project's own API tests caught (TestClient genuinely
        # exercises FastAPI's thread dispatch, unlike calling the store
        # directly). Safe here because `AdmissionController` (Module 6.5)
        # already bounds this service to sequential access by default
        # (max_concurrent_requests: 1); true concurrent multi-threaded
        # writes would need an explicit lock or WAL mode, out of scope.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExtractionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save(self, record: ExtractionRecord) -> None:
        self._conn.execute(
            "INSERT INTO extractions (request_id, trace_id, schema_name, raw_input, "
            "extracted_fields, confidence, needs_review, validation_error, used_repair_retry, "
            "latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.request_id,
                record.trace_id,
                record.schema_name,
                record.raw_input,
                json.dumps(record.extracted_fields),
                record.confidence,
                int(record.needs_review),
                record.validation_error,
                int(record.used_repair_retry),
                record.latency_ms,
            ),
        )
        self._conn.commit()

    def get(self, request_id: str) -> ExtractionRecord | None:
        row = self._conn.execute(
            "SELECT request_id, trace_id, schema_name, raw_input, extracted_fields, confidence, "
            "needs_review, validation_error, used_repair_retry, latency_ms, created_at "
            "FROM extractions WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_low_confidence(self, *, limit: int = 50) -> list[ExtractionRecord]:
        rows = self._conn.execute(
            "SELECT request_id, trace_id, schema_name, raw_input, extracted_fields, confidence, "
            "needs_review, validation_error, used_repair_retry, latency_ms, created_at "
            "FROM extractions WHERE needs_review = 1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self, *, limit: int = 100) -> list[ExtractionRecord]:
        rows = self._conn.execute(
            "SELECT request_id, trace_id, schema_name, raw_input, extracted_fields, confidence, "
            "needs_review, validation_error, used_repair_retry, latency_ms, created_at "
            "FROM extractions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> ExtractionRecord:
        return ExtractionRecord(
            request_id=row[0],
            trace_id=row[1],
            schema_name=row[2],
            raw_input=row[3],
            extracted_fields=json.loads(row[4]),
            confidence=row[5],
            needs_review=bool(row[6]),
            validation_error=row[7],
            used_repair_retry=bool(row[8]),
            latency_ms=row[9],
            created_at=row[10],
        )
