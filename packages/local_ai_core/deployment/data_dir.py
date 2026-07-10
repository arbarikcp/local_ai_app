"""Data directory layout (theory doc §11) — real subdirectories under
`app.data_dir`, one per this repo's existing SQLite store (Module 8.5's
`SessionStore`, Module 14's `AuditLog`, Module 19's `AdapterRegistry`,
Module 21's `EvalFeedbackStore`) - the first time any of those stores gets
a real, consistent on-disk home instead of a bespoke path per demo script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.config import AppConfig


@dataclass(frozen=True)
class DataDirectoryLayout:
    base_dir: Path
    sessions_db: Path
    audit_db: Path
    adapters_db: Path
    eval_feedback_db: Path
    backups_dir: Path


def resolve_data_dir(config: AppConfig) -> Path:
    return Path(config.app.data_dir).expanduser()


def ensure_data_dir_layout(config: AppConfig) -> DataDirectoryLayout:
    base_dir = resolve_data_dir(config)
    layout = DataDirectoryLayout(
        base_dir=base_dir,
        sessions_db=base_dir / "sessions" / "sessions.db",
        audit_db=base_dir / "audit" / "audit.db",
        adapters_db=base_dir / "adapters" / "adapters.db",
        eval_feedback_db=base_dir / "eval_feedback" / "eval_feedback.db",
        backups_dir=base_dir / "backups",
    )
    for path in (layout.sessions_db, layout.audit_db, layout.adapters_db, layout.eval_feedback_db):
        path.parent.mkdir(parents=True, exist_ok=True)
    layout.backups_dir.mkdir(parents=True, exist_ok=True)
    return layout
