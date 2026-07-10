"""Patch scope and line-count validation (ARCHITECTURE.md "Patch scope
and line-count validation") — closes two real gaps confirmed by reading
Module 17's `patch_tools.py`: its hunk-header regex's `,N` count groups are
non-capturing (a hunk whose stated line count doesn't match its actual
body isn't flagged), and nothing anywhere checks a patch's target file
against what the request was actually about (curriculum's own "model
changes unrelated files" failure case). Both run before `apply_patch()` is
ever called - real, deterministic checks on top of Module 17's real,
unchanged parser, not a replacement for it.
"""

from __future__ import annotations

import re

from local_ai_agents.tools.patch_tools import ParsedPatch, validate_patch_format

_HUNK_HEADER_FULL_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", re.MULTILINE)


class PatchScopeError(Exception):
    def __init__(self, actual_file_path: str, expected_file_path: str) -> None:
        super().__init__(
            f"patch targets {actual_file_path!r} but the request was about {expected_file_path!r} - refusing to apply"
        )
        self.actual_file_path = actual_file_path
        self.expected_file_path = expected_file_path


class PatchLineCountError(Exception):
    pass


def validate_patch_scope(parsed: ParsedPatch, expected_file_path: str) -> None:
    if parsed.file_path != expected_file_path:
        raise PatchScopeError(parsed.file_path, expected_file_path)


def validate_hunk_line_counts(patch_text: str) -> None:
    """Re-parses each `@@ -start,count +start,count @@` header's own
    claimed line counts (deliberately ignored by `patch_tools.py`'s
    parser) and checks them against the hunk body's real line count. A
    missing count segment (bare `@@ -N +N @@`, meaning "1 line," valid
    unified-diff shorthand) is not checked - only a count that's actually
    present and wrong is an error.
    """
    parsed: ParsedPatch = validate_patch_format(patch_text)
    headers = _HUNK_HEADER_FULL_RE.findall(patch_text)

    if len(headers) != len(parsed.hunks):
        raise PatchLineCountError(
            f"found {len(headers)} hunk header(s) but {len(parsed.hunks)} parsed hunk(s) - patch is malformed"
        )

    for (old_count_str, new_count_str), hunk in zip(headers, parsed.hunks):
        if old_count_str and int(old_count_str) != len(hunk.old_lines):
            raise PatchLineCountError(
                f"hunk at line {hunk.old_start} claims {old_count_str} old line(s) but its body has "
                f"{len(hunk.old_lines)}"
            )
        if new_count_str and int(new_count_str) != len(hunk.new_lines):
            raise PatchLineCountError(
                f"hunk at line {hunk.old_start} claims {new_count_str} new line(s) but its body has "
                f"{len(hunk.new_lines)}"
            )
