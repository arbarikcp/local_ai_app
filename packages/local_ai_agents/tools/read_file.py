"""read_file tool (curriculum's required tools list) - reads a specific
line range of a sandboxed file. Real, useful for the "read relevant
files" architecture step without loading an entire (possibly large) file
into context.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.sandbox import resolve_within_sandbox


class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1, max_length=200)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


def read_file_lines(allowed_base: Path, relative_path: str, start_line: int = 1, end_line: int | None = None) -> str:
    target = resolve_within_sandbox(allowed_base, relative_path)
    lines = target.read_text(encoding="utf-8").splitlines()
    end = end_line if end_line is not None else len(lines)
    if start_line > end:
        raise ValueError(f"start_line ({start_line}) must be <= end_line ({end})")
    return "\n".join(lines[start_line - 1 : end])


def make_read_file_tool(allowed_base: Path) -> Tool:
    async def handler(args: ReadFileArgs) -> str:
        return read_file_lines(allowed_base, args.path, args.start_line, args.end_line)

    return Tool(
        name="read_file",
        description="Read a line range from a sandboxed file.",
        args_model=ReadFileArgs,
        handler=handler,
    )
