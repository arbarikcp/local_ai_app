"""Real regex-based secrets/credential detection (theory doc §9) — a
sibling to Module 21's `pii_redaction.py`, but deliberately detection-only,
no redact-and-continue: a secret partially revealed (e.g. `sk-...7f2a`) is
still a leak, unlike a PII field where showing "[EMAIL]" is genuinely safe.
Any match here is a hard signal upstream, not something to mask and log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\bBearer [A-Za-z0-9\-_.=]{10,}\b"),
    "generic_api_key": re.compile(r"(?i)\bapi[_-]?key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}"),
}


@dataclass(frozen=True)
class SecretsScanResult:
    matches: dict[str, int] = field(default_factory=dict)

    @property
    def total_matches(self) -> int:
        return sum(self.matches.values())

    @property
    def found_secrets(self) -> bool:
        return self.total_matches > 0


def scan_for_secrets(text: str) -> SecretsScanResult:
    matches: dict[str, int] = {}
    for category, pattern in _PATTERNS.items():
        count = len(pattern.findall(text))
        if count:
            matches[category] = count
    return SecretsScanResult(matches=matches)
