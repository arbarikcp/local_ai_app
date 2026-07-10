"""DocStorage — real, persistent SQLite storage for per-document ingestion
status and per-page routing/extraction results (ARCHITECTURE.md "Storage
schema"). Same idiom as Project 2's `rag_metadata_store.py` and Project 1's
`extraction_storage.py`, including `check_same_thread=False` applied from
the start (Project 1's own real bug fix under FastAPI's `TestClient`
threading).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS page_analyses (
    page_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    route TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    extracted_fields TEXT,
    confidence TEXT,
    needs_review INTEGER,
    quarantine_reason TEXT
);
"""

VALID_DOCUMENT_STATUSES = ("ingested", "failed")


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_path: str
    page_count: int
    status: str
    ingested_at: str | None = None


@dataclass(frozen=True)
class PageAnalysisRecord:
    page_id: str
    doc_id: str
    page_number: int
    route: str
    route_reason: str
    extracted_text: str
    extracted_fields: dict[str, object] | None = None
    confidence: str | None = None
    needs_review: bool = False
    quarantine_reason: str | None = None


class DocStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DocStorage:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save_document(self, record: DocumentRecord) -> None:
        if record.status not in VALID_DOCUMENT_STATUSES:
            raise ValueError(f"status must be one of {VALID_DOCUMENT_STATUSES}, got {record.status!r}")
        self._conn.execute(
            "INSERT INTO documents (doc_id, source_path, page_count, status) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET source_path=excluded.source_path, "
            "page_count=excluded.page_count, status=excluded.status, ingested_at=datetime('now')",
            (record.doc_id, record.source_path, record.page_count, record.status),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT doc_id, source_path, page_count, status, ingested_at FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def save_page_analysis(self, record: PageAnalysisRecord) -> None:
        extracted_fields_json = json.dumps(record.extracted_fields) if record.extracted_fields is not None else None
        self._conn.execute(
            "INSERT INTO page_analyses (page_id, doc_id, page_number, route, route_reason, "
            "extracted_text, extracted_fields, confidence, needs_review, quarantine_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET route=excluded.route, "
            "route_reason=excluded.route_reason, extracted_text=excluded.extracted_text, "
            "extracted_fields=excluded.extracted_fields, confidence=excluded.confidence, "
            "needs_review=excluded.needs_review, quarantine_reason=excluded.quarantine_reason",
            (
                record.page_id,
                record.doc_id,
                record.page_number,
                record.route,
                record.route_reason,
                record.extracted_text,
                extracted_fields_json,
                record.confidence,
                int(record.needs_review),
                record.quarantine_reason,
            ),
        )
        self._conn.commit()

    def get_page_analysis(self, page_id: str) -> PageAnalysisRecord | None:
        row = self._conn.execute(
            "SELECT page_id, doc_id, page_number, route, route_reason, extracted_text, "
            "extracted_fields, confidence, needs_review, quarantine_reason "
            "FROM page_analyses WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_page_analysis(row)

    def list_page_analyses(self, doc_id: str) -> list[PageAnalysisRecord]:
        rows = self._conn.execute(
            "SELECT page_id, doc_id, page_number, route, route_reason, extracted_text, "
            "extracted_fields, confidence, needs_review, quarantine_reason "
            "FROM page_analyses WHERE doc_id = ? ORDER BY page_number ASC",
            (doc_id,),
        ).fetchall()
        return [self._row_to_page_analysis(row) for row in rows]

    @staticmethod
    def _row_to_document(row: tuple) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row[0],
            source_path=row[1],
            page_count=row[2],
            status=row[3],
            ingested_at=row[4],
        )

    @staticmethod
    def _row_to_page_analysis(row: tuple) -> PageAnalysisRecord:
        return PageAnalysisRecord(
            page_id=row[0],
            doc_id=row[1],
            page_number=row[2],
            route=row[3],
            route_reason=row[4],
            extracted_text=row[5],
            extracted_fields=json.loads(row[6]) if row[6] is not None else None,
            confidence=row[7],
            needs_review=bool(row[8]),
            quarantine_reason=row[9],
        )
