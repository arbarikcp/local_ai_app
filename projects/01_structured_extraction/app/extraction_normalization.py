"""Text normalization before extraction (ARCHITECTURE.md "Data flow
through one request", step 2). Nothing purpose-built for document
normalization exists anywhere in the repo (confirmed by survey). The
whitespace-collapse regexes mirror Module 20's `optimization.prompt_compression`
(same three patterns: trailing spaces, repeated spaces, excess blank
lines) rather than importing its private helper directly - that function
is an internal implementation detail of a different module's public API,
not something this project should couple to. Deliberately does NOT reuse
prompt_compression's duplicate-line removal: a real document can
legitimately repeat a line (e.g. two identical invoice line items), and
normalization must not silently drop real content the way prompt
compression is allowed to.
"""

from __future__ import annotations

import re

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+\n")
_MULTIPLE_SPACES_RE = re.compile(r" {2,}")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _collapse_whitespace(text: str) -> str:
    text = _TRAILING_SPACES_RE.sub("\n", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    return _MULTIPLE_BLANK_LINES_RE.sub("\n\n", text)


class TextTooLongError(Exception):
    def __init__(self, actual_length: int, max_chars: int) -> None:
        super().__init__(f"normalized text is {actual_length} chars, exceeds the {max_chars}-char limit")
        self.actual_length = actual_length
        self.max_chars = max_chars


def normalize_text(text: str, *, max_chars: int | None = None) -> str:
    """Strips control characters (keeping newlines/tabs), collapses
    whitespace, and trims leading/trailing whitespace. Raises
    `TextTooLongError` rather than silently truncating when `max_chars` is
    given and exceeded - a caller must decide how to handle an over-length
    document, never have it quietly cut off mid-sentence.
    """
    stripped = _CONTROL_CHAR_RE.sub("", text)
    collapsed = _collapse_whitespace(stripped).strip()

    if max_chars is not None and len(collapsed) > max_chars:
        raise TextTooLongError(len(collapsed), max_chars)

    return collapsed
