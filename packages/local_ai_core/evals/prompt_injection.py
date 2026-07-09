"""Prompt injection detection for retrieved documents (theory doc §11) - a
real, if necessarily incomplete, regex-based screen for common injection
phrasings that might appear *inside a retrieved document* rather than in
the user's own message (curriculum's own threat model: "a malicious
document changes model behavior"). A pattern screen, not a guarantee: a
sufficiently rephrased injection defeats it - documented honestly, the
same standard applied to every heuristic in this course.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |the )?previous instructions",
    r"disregard (all |the )?(above|previous)",
    r"you are now",
    r"reveal (the |your )?system prompt",
    r"forget (all |everything )?(you (were|have been) told|your instructions)",
    r"act as (if you (are|were)|an?)\b",
    r"new instructions?:",
]

_COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]


def detect_prompt_injection_patterns(text: str) -> list[str]:
    """Returns the list of pattern strings (from `INJECTION_PATTERNS`) that
    matched, in check order. An empty list means no known pattern matched
    - not the same as "safe", since this is a screen, not a guarantee.
    """
    return [
        pattern_str
        for pattern_str, compiled in zip(INJECTION_PATTERNS, _COMPILED_PATTERNS)
        if compiled.search(text)
    ]


def contains_prompt_injection(text: str) -> bool:
    return len(detect_prompt_injection_patterns(text)) > 0
