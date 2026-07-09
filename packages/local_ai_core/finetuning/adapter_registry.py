"""AdapterRegistry — SQLite-backed LoRA adapter metadata (theory doc §10).
Same pattern as Module 8.5's `SessionStore` and Module 14's `AuditLog`: no
server, no external dependency, Python's stdlib sqlite3 module.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS adapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    base_model TEXT NOT NULL,
    rank INTEGER NOT NULL,
    alpha INTEGER NOT NULL,
    target_modules TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class AdapterRecord:
    name: str
    base_model: str
    rank: int
    alpha: int
    target_modules: list[str]
    dataset_hash: str
    file_path: str
    created_at: str | None = None


class AdapterRegistry:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AdapterRegistry:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def register(self, record: AdapterRecord) -> None:
        self._conn.execute(
            "INSERT INTO adapters (name, base_model, rank, alpha, target_modules, dataset_hash, file_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.name,
                record.base_model,
                record.rank,
                record.alpha,
                ",".join(record.target_modules),
                record.dataset_hash,
                record.file_path,
            ),
        )
        self._conn.commit()

    def get(self, name: str) -> AdapterRecord | None:
        row = self._conn.execute(
            "SELECT name, base_model, rank, alpha, target_modules, dataset_hash, file_path, created_at "
            "FROM adapters WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_adapters(self) -> list[AdapterRecord]:
        rows = self._conn.execute(
            "SELECT name, base_model, rank, alpha, target_modules, dataset_hash, file_path, created_at "
            "FROM adapters ORDER BY created_at ASC"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, name: str) -> None:
        self._conn.execute("DELETE FROM adapters WHERE name = ?", (name,))
        self._conn.commit()

    @staticmethod
    def _row_to_record(row: tuple) -> AdapterRecord:
        return AdapterRecord(
            name=row[0],
            base_model=row[1],
            rank=row[2],
            alpha=row[3],
            target_modules=row[4].split(",") if row[4] else [],
            dataset_hash=row[5],
            file_path=row[6],
            created_at=row[7],
        )
