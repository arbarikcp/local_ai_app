"""RAG data-poisoning defense (theory doc §7) — every document is screened
for injection patterns before ingestion, regardless of declared source
trust. A "trusted" source is still a real threat curriculum's own threat
model names (a compromised web page, a tampered upload) - `SourceTrust`
is recorded for audit purposes, not used to skip the scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from local_ai_core.evals.prompt_injection import detect_prompt_injection_patterns


class SourceTrust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class IngestionDecision:
    allowed: bool
    source_trust: SourceTrust
    reason: str
    flagged_patterns: list[str]


def screen_document_for_ingestion(text: str, *, source_trust: SourceTrust) -> IngestionDecision:
    flagged_patterns = detect_prompt_injection_patterns(text)

    if flagged_patterns:
        return IngestionDecision(
            allowed=False,
            source_trust=source_trust,
            reason=f"quarantined: {len(flagged_patterns)} injection pattern(s) matched",
            flagged_patterns=flagged_patterns,
        )

    return IngestionDecision(
        allowed=True,
        source_trust=source_trust,
        reason="no injection patterns matched",
        flagged_patterns=[],
    )
