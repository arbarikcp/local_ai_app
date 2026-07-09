"""Context budgeting and packing (theory doc "Context packing strategy",
"Lost-in-the-middle mitigation"). Implements the curriculum's exact budget
shape and packs candidates by relevance, source diversity, and token
budget. Same discipline Module 1 established for token counting: this
heuristic (~1.3 tokens/word) is explicitly labeled, never trusted as an
exact count a real tokenizer would produce.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_rag.embeddings.embedder import SearchResult

_HEURISTIC_TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    words = text.split()
    return max(1, round(len(words) * _HEURISTIC_TOKENS_PER_WORD))


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int
    reserved_for_system: int
    reserved_for_question: int
    reserved_for_answer: int

    def __post_init__(self) -> None:
        for name in ("max_context_tokens", "reserved_for_system", "reserved_for_question", "reserved_for_answer"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def available_for_chunks(self) -> int:
        """Never negative - an over-reserved budget reads as "no room for
        chunks," not a silent negative number a caller might misuse.
        """
        remaining = self.max_context_tokens - self.reserved_for_system - self.reserved_for_question - self.reserved_for_answer
        return max(remaining, 0)


def pack_context(
    candidates: list[SearchResult],
    budget: ContextBudget,
    *,
    max_chunks_per_source: int = 2,
    source_key: str = "doc_id",
) -> list[SearchResult]:
    """Packs `candidates` (assumed already ranked by relevance - rerank
    first) within `budget.available_for_chunks`, in order: relevance
    (input order), source diversity (skip a candidate once
    `max_chunks_per_source` from its source are already packed, so one
    dominant document can't crowd out every other source), then token
    budget (skip, don't stop - a later, shorter candidate may still fit).
    """
    packed: list[SearchResult] = []
    used_tokens = 0
    count_by_source: dict[str, int] = {}

    for candidate in candidates:
        source = candidate.metadata.get(source_key, candidate.doc_id)
        if count_by_source.get(source, 0) >= max_chunks_per_source:
            continue
        cost = estimate_tokens(candidate.text)
        if used_tokens + cost > budget.available_for_chunks:
            continue
        packed.append(candidate)
        used_tokens += cost
        count_by_source[source] = count_by_source.get(source, 0) + 1

    return packed


def order_for_generation(packed: list[SearchResult]) -> list[SearchResult]:
    """Lost-in-the-middle mitigation (theory doc §13): alternately place
    the next-most-relevant remaining chunk at the start, then the end, of
    the output - the highest-relevance chunks end up at both edges of the
    context, the weakest in the middle, matching the empirically weak spot
    for long-context attention rather than ignoring it.
    """
    ordered: list[SearchResult | None] = [None] * len(packed)
    left, right = 0, len(packed) - 1
    for i, candidate in enumerate(packed):
        if i % 2 == 0:
            ordered[left] = candidate
            left += 1
        else:
            ordered[right] = candidate
            right -= 1
    return [c for c in ordered if c is not None]
