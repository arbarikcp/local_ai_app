"""Time-aware retrieval (theory doc §15) - combine similarity with an
exponential recency decay based on a `created_at` metadata timestamp, so a
genuinely older-but-still-relevant document can still outrank a barely-
relevant recent one (decay boosts, never fully overrides, relevance).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from local_ai_rag.embeddings.embedder import SearchResult


def _age_days(created_at: str, now: datetime) -> float:
    created = datetime.fromisoformat(created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max((now - created).total_seconds() / 86400, 0.0)


def apply_recency_boost(
    results: list[SearchResult],
    *,
    metadata_key: str = "created_at",
    half_life_days: float = 30.0,
    now: datetime | None = None,
) -> list[SearchResult]:
    """`score' = score * 2^(-age_days / half_life_days)`. A document
    exactly `half_life_days` old has its score halved; a document with no
    `metadata_key` at all is left unboosted (treated as ageless, not
    penalized for missing metadata a caller never attached).
    """
    now = now or datetime.now(timezone.utc)
    boosted = []
    for result in results:
        created_at = result.metadata.get(metadata_key)
        if created_at is None:
            boosted.append(result)
            continue
        age_days = _age_days(created_at, now)
        decay = 2 ** (-age_days / half_life_days) if half_life_days > 0 else 1.0
        boosted.append(replace(result, score=result.score * decay))
    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted
