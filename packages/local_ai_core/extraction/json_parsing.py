"""Loose JSON parsing for model output (theory doc Gotchas: models may
return markdown-wrapped JSON or add comments inside it).

Intentionally a small, self-contained duplication of
scripts/module_03/scorers/json_validity.py's markdown-fence-stripping
logic, NOT a cross-boundary import - packages/ must not depend on
scripts/module_NN/ (the reverse is the expected, common direction), so this
~15-line helper is duplicated rather than inverting that layering.
"""

from __future__ import annotations

import json
import re
from typing import Any

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_markdown_fence(text: str) -> str:
    """Remove a wrapping ```json ... ``` or ``` ... ``` fence, if present."""
    match = _MARKDOWN_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def try_parse_json(text: str) -> Any | None:
    """Parse JSON after stripping a markdown fence; return None on failure."""
    candidate = strip_markdown_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
