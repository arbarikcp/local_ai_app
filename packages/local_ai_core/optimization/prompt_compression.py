"""Real, deterministic, non-LLM prompt compression (theory doc §3) —
whitespace collapsing and exact-duplicate consecutive line removal.
Distinct from Module 8.5's `summarizer.py`, which compresses conversation
history via an LLM call (lossy, needs a model); this needs no model at
all, at the cost of a much smaller (but real, measurable) reduction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEURISTIC_TOKENS_PER_WORD = 1.3
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_MULTIPLE_SPACES = re.compile(r" {2,}")


def _estimate_tokens(text: str) -> int:
    words = text.split()
    return max(0, round(len(words) * _HEURISTIC_TOKENS_PER_WORD))


def _collapse_whitespace(text: str) -> str:
    text = _TRAILING_SPACES.sub("\n", text)
    text = _MULTIPLE_SPACES.sub(" ", text)
    return _MULTIPLE_BLANK_LINES.sub("\n\n", text)


def _drop_duplicate_consecutive_lines(text: str) -> str:
    lines = text.split("\n")
    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line and line.strip() != "":
            continue
        deduped.append(line)
    return "\n".join(deduped)


@dataclass(frozen=True)
class CompressionResult:
    original_text: str
    compressed_text: str
    original_token_estimate: int
    compressed_token_estimate: int

    @property
    def reduction_ratio(self) -> float:
        """Fraction of estimated tokens removed - 0.0 when there was
        nothing to compress, never negative even if compression somehow
        left the text unchanged.
        """
        if self.original_token_estimate == 0:
            return 0.0
        removed = self.original_token_estimate - self.compressed_token_estimate
        return max(0.0, removed / self.original_token_estimate)


def compress_prompt(text: str) -> CompressionResult:
    compressed = _collapse_whitespace(text)
    compressed = _drop_duplicate_consecutive_lines(compressed)
    return CompressionResult(
        original_text=text,
        compressed_text=compressed,
        original_token_estimate=_estimate_tokens(text),
        compressed_token_estimate=_estimate_tokens(compressed),
    )
