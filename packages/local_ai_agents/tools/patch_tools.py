"""Patch proposal, validation, and application (curriculum's required
tools: `propose_patch`, `apply_patch`; theory doc §9-11, §13 "Code
hallucination"). `propose_patch` is LLM-driven (`FakeRuntime`-backed);
`validate_patch_format` and `apply_patch` are real, deterministic unified-
diff parsing/application - a proposed patch is never trusted as valid
just because an LLM produced text that looks like a diff, and `apply_patch`
refuses to touch a file whose actual content doesn't match what the patch
claims to be replacing (a hallucinated hunk is rejected, not silently
misapplied).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_agents.tools.sandbox import resolve_within_sandbox

PATCH_PROMPT_TEMPLATE = """You are proposing a code fix as a unified diff.

Instruction:
{instruction}

Relevant file contents:
{file_contents}

Respond with only a valid unified diff (--- / +++ / @@ hunks), nothing else."""


class PatchFormatError(Exception):
    pass


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


@dataclass(frozen=True)
class PatchHunk:
    old_start: int
    old_lines: list[str]
    new_lines: list[str]


@dataclass(frozen=True)
class ParsedPatch:
    file_path: str
    hunks: list[PatchHunk]


async def propose_patch(instruction: str, file_contents: dict[str, str], runtime: LLMRuntime, model: str) -> str:
    contents_block = "\n\n".join(f"--- {path} ---\n{text}" for path, text in file_contents.items())
    prompt = PATCH_PROMPT_TEMPLATE.format(instruction=instruction, file_contents=contents_block)
    response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
    return response.text.strip()


def validate_patch_format(patch_text: str) -> ParsedPatch:
    """A deliberately strict, minimal unified-diff subset: one file header
    (`--- old` / `+++ new`), one or more `@@` hunks, each hunk body made
    only of context (` `), removed (`-`), and added (`+`) lines. Anything
    else raises `PatchFormatError` rather than being guessed at.
    """
    lines = patch_text.splitlines()
    file_path: str | None = None
    hunks: list[PatchHunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("--- "):
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise PatchFormatError("Expected a '+++ ' line immediately after the '--- ' line")
            file_path = lines[i][4:].strip()
            if file_path.startswith("b/"):
                file_path = file_path[2:]
            i += 1
            continue

        match = _HUNK_HEADER_RE.match(line)
        if match:
            old_start = int(match.group(1))
            i += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                hunk_line = lines[i]
                if hunk_line.startswith("-"):
                    old_lines.append(hunk_line[1:])
                elif hunk_line.startswith("+"):
                    new_lines.append(hunk_line[1:])
                elif hunk_line.startswith(" "):
                    old_lines.append(hunk_line[1:])
                    new_lines.append(hunk_line[1:])
                elif hunk_line.strip() == "":
                    pass
                else:
                    raise PatchFormatError(f"Unexpected line inside a hunk: {hunk_line!r}")
                i += 1
            hunks.append(PatchHunk(old_start=old_start, old_lines=old_lines, new_lines=new_lines))
            continue

        i += 1

    if file_path is None:
        raise PatchFormatError("Patch is missing a '--- '/'+++ ' file header")
    if not hunks:
        raise PatchFormatError("Patch has no '@@' hunks")
    return ParsedPatch(file_path=file_path, hunks=hunks)


def apply_patch(allowed_base: Path, patch_text: str) -> str:
    """Applies a validated patch to a sandboxed file - a real find-and-
    replace of each hunk's old-line block with its new-line block, never a
    no-op simulation. Every hunk's claimed old content is checked against
    the file's real current content before any write happens; a mismatch
    (the exact signature of a hallucinated or stale patch) raises
    `PatchFormatError` instead of corrupting the file.
    """
    parsed = validate_patch_format(patch_text)
    target = resolve_within_sandbox(allowed_base, parsed.file_path)
    result_lines = target.read_text(encoding="utf-8").splitlines()

    # Apply from the bottom up so line numbers named by earlier hunks aren't
    # invalidated by insertions/deletions from later ones.
    for hunk in sorted(parsed.hunks, key=lambda h: h.old_start, reverse=True):
        start = hunk.old_start - 1
        end = start + len(hunk.old_lines)
        actual = result_lines[start:end]
        if actual != hunk.old_lines:
            raise PatchFormatError(
                f"Patch context mismatch at line {hunk.old_start}: "
                f"expected {hunk.old_lines!r}, found {actual!r} - refusing to apply"
            )
        result_lines[start:end] = hunk.new_lines

    target.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    return str(target.resolve().relative_to(allowed_base.resolve()))
