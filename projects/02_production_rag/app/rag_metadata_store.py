"""RagMetadataStore — real, persistent SQLite document metadata and
ingestion-status tracking, plus a query log (ARCHITECTURE.md "Storage
schema"). `IncrementalIndexer` (Module 12) has real diff logic but keeps
its manifest in memory only - nothing survives a process restart there.
Same idiom as Project 1's `extraction_storage.py`, including its real
`check_same_thread=False` fix applied from the start here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT,
    title TEXT,
    status TEXT NOT NULL,
    content_hash TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS query_log (
    query_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    citation_count INTEGER NOT NULL,
    verified_citation_count INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

VALID_STATUSES = ("ingested", "quarantined", "unchanged", "failed")


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_path: str | None
    title: str | None
    status: str
    content_hash: str | None
    chunk_count: int
    quarantine_reason: str | None = None
    ingested_at: str | None = None


@dataclass(frozen=True)
class QueryLogRecord:
    query_id: str
    question: str
    answer_text: str
    citation_count: int
    verified_citation_count: int
    latency_ms: float
    created_at: str | None = None


class RagMetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RagMetadataStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save_document(self, record: DocumentRecord) -> None:
        if record.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {record.status!r}")
        self._conn.execute(
            "INSERT INTO documents (doc_id, source_path, title, status, content_hash, "
            "chunk_count, quarantine_reason) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET source_path=excluded.source_path, "
            "title=excluded.title, status=excluded.status, content_hash=excluded.content_hash, "
            "chunk_count=excluded.chunk_count, quarantine_reason=excluded.quarantine_reason, "
            "ingested_at=datetime('now')",
            (
                record.doc_id,
                record.source_path,
                record.title,
                record.status,
                record.content_hash,
                record.chunk_count,
                record.quarantine_reason,
            ),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT doc_id, source_path, title, status, content_hash, chunk_count, "
            "quarantine_reason, ingested_at FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def delete_document(self, doc_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_documents(self, *, status: str | None = None) -> list[DocumentRecord]:
        if status is not None:
            rows = self._conn.execute(
                "SELECT doc_id, source_path, title, status, content_hash, chunk_count, "
                "quarantine_reason, ingested_at FROM documents WHERE status = ? ORDER BY ingested_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT doc_id, source_path, title, status, content_hash, chunk_count, "
                "quarantine_reason, ingested_at FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def log_query(self, record: QueryLogRecord) -> None:
        self._conn.execute(
            "INSERT INTO query_log (query_id, question, answer_text, citation_count, "
            "verified_citation_count, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.query_id,
                record.question,
                record.answer_text,
                record.citation_count,
                record.verified_citation_count,
                record.latency_ms,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_document(row: tuple) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row[0],
            source_path=row[1],
            title=row[2],
            status=row[3],
            content_hash=row[4],
            chunk_count=row[5],
            quarantine_reason=row[6],
            ingested_at=row[7],
        )
