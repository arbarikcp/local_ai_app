"""SQLite backup and restore (theory doc §12) — real `sqlite3` `.backup()`
API, the correct way to back up a live SQLite database (it takes a
consistent snapshot even mid-write, unlike a naive file copy that could
catch a half-written page). Proven across an actual backup-then-restore-
then-read round trip, not asserted from the API's own documentation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backup_sqlite_db(source_path: str | Path, backup_dir: str | Path) -> Path:
    source_path = Path(source_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_dir / f"{source_path.stem}_{timestamp}.db"

    source_conn = sqlite3.connect(str(source_path))
    dest_conn = sqlite3.connect(str(backup_path))
    try:
        source_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        source_conn.close()

    return backup_path


def restore_sqlite_db(backup_path: str | Path, destination_path: str | Path) -> None:
    backup_path = Path(backup_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    backup_conn = sqlite3.connect(str(backup_path))
    dest_conn = sqlite3.connect(str(destination_path))
    try:
        backup_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        backup_conn.close()


def list_backups(backup_dir: str | Path) -> list[Path]:
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        return []
    return sorted(backup_dir.glob("*.db"))
