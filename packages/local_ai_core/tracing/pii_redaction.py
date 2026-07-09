"""Real regex-based PII redaction (theory doc §5) — emails, phone numbers,
credit-card-shaped numbers, and SSN-shaped numbers. Pattern matching, not a
model call: cheap, deterministic, and exactly what a logging pipeline needs
to run on every request without adding latency. Reports a real per-category
count rather than silently dropping evidence that redaction happened -
"0 redactions" and "redaction never ran" must never look the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Order matters: credit_card's digit-run pattern would also match inside an
# SSN or a phone number, so the more specific patterns run first and their
# matches are masked before the broader digit-run pattern gets a chance.
_PATTERN_ORDER = ("email", "ssn", "phone", "credit_card")


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    redaction_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        return sum(self.redaction_counts.values())


def redact_pii(text: str) -> RedactionResult:
    redacted = text
    counts: dict[str, int] = {}

    for category in _PATTERN_ORDER:
        pattern = _PATTERNS[category]
        matches = pattern.findall(redacted)
        if matches:
            counts[category] = len(pattern.findall(redacted))
            redacted = pattern.sub(f"[{category.upper()}]", redacted)

    return RedactionResult(redacted_text=redacted, redaction_counts=counts)
