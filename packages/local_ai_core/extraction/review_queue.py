"""ReviewQueue — the last resort in the reliability ladder (theory doc §8):
low-confidence extractions land here rather than being silently persisted
or silently discarded.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .confidence import ConfidenceLevel


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    extracted_fields: dict[str, Any]
    confidence: ConfidenceLevel
    reason: str
    source_text: str
    queued_at: datetime


@dataclass(frozen=True)
class ReviewResolution:
    item_id: str
    approved: bool
    corrected_fields: dict[str, Any] | None
    resolved_at: datetime


class ReviewQueue:
    """In-memory human review queue. A real deployment would persist this
    (SQLite, per curriculum.md §18's production pipeline); persistence is
    application-specific and out of scope for this reusable component
    (same boundary the pipeline itself draws for "persist result").
    """

    def __init__(self) -> None:
        self._pending: dict[str, ReviewItem] = {}
        self._resolved: dict[str, ReviewResolution] = {}

    def enqueue(
        self, extracted_fields: dict[str, Any], confidence: ConfidenceLevel, reason: str, source_text: str
    ) -> ReviewItem:
        item = ReviewItem(
            item_id=str(uuid.uuid4()),
            extracted_fields=extracted_fields,
            confidence=confidence,
            reason=reason,
            source_text=source_text,
            queued_at=datetime.now(timezone.utc),
        )
        self._pending[item.item_id] = item
        return item

    def list_pending(self) -> list[ReviewItem]:
        return sorted(self._pending.values(), key=lambda item: item.queued_at)

    def resolve(self, item_id: str, *, approved: bool, corrected_fields: dict[str, Any] | None = None) -> ReviewResolution:
        if item_id not in self._pending:
            raise KeyError(f"No pending review item with id {item_id!r}")
        del self._pending[item_id]
        resolution = ReviewResolution(
            item_id=item_id,
            approved=approved,
            corrected_fields=corrected_fields,
            resolved_at=datetime.now(timezone.utc),
        )
        self._resolved[item_id] = resolution
        return resolution

    def get_resolution(self, item_id: str) -> ReviewResolution | None:
        return self._resolved.get(item_id)

    def __len__(self) -> int:
        return len(self._pending)
