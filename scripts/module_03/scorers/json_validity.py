"""JSON structural validity scoring.

This is deliberately *structural* validity only — whether the model produced
parseable JSON matching a shape — not semantic correctness of the content.
Module 8 (structured output and extraction) builds the full validation
pipeline (constrained decoding, Pydantic schema validation, repair retries);
this module only needs enough to score a benchmark run.
"""

from __future__ import annotations

import json
import re
from typing import Any

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_markdown_fence(text: str) -> str:
    """Remove a wrapping ```json ... ``` or ``` ... ``` fence, if present.

    Small local models frequently wrap JSON in a markdown fence despite being
    told not to (docs/modules/01_local_llm_systems_thinking.md §11) — this
    normalizes that specific, extremely common failure before parsing.
    """
    match = _MARKDOWN_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def try_parse_json(text: str) -> Any | None:
    """Parse JSON after stripping a markdown fence; return None on failure."""
    candidate = strip_markdown_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def is_valid_json(text: str) -> bool:
    return try_parse_json(text) is not None


def has_required_keys(text: str, required_keys: list[str]) -> bool:
    """True if the text parses as a JSON object containing every required key."""
    parsed = try_parse_json(text)
    if not isinstance(parsed, dict):
        return False
    return all(key in parsed for key in required_keys)


def invalid_json_rate(predictions: list[str]) -> float:
    """Fraction of predictions that do NOT parse as valid JSON.

    This is the "invalid JSON rate" reliability metric from the benchmark
    dimensions table — lower is better, unlike the other scorers here.
    """
    if not predictions:
        return 0.0
    invalid = sum(1 for p in predictions if not is_valid_json(p))
    return invalid / len(predictions)
