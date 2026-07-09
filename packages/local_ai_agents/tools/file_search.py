"""File search tool (Lab 3) - curriculum's own `SearchFilesArgs` example.
Real path-containment via `sandbox.py`: `root_path` is resolved *relative
to* the tool's `allowed_base` (never the process's CWD), and traversal
outside it raises `PathTraversalError` before any filesystem walk happens.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.sandbox import resolve_within_sandbox


class SearchFilesArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    root_path: str = Field(default=".")
    max_results: int = Field(default=10, ge=1, le=50)


def _content_matches(path: Path, query: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return query.lower() in text.lower()


def search_files(allowed_base: Path, query: str, root_path: str = ".", max_results: int = 10) -> list[str]:
    root = resolve_within_sandbox(allowed_base, root_path)
    if not root.is_dir():
        raise NotADirectoryError(f"root_path is not a directory: {root_path!r}")

    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if query.lower() in path.name.lower() or _content_matches(path, query):
            matches.append(str(path.resolve().relative_to(allowed_base.resolve())))
        if len(matches) >= max_results:
            break
    return matches


def make_file_search_tool(allowed_base: Path) -> Tool:
    async def handler(args: SearchFilesArgs) -> list[str]:
        return search_files(allowed_base, args.query, args.root_path, args.max_results)

    return Tool(
        name="search_files",
        description="Search for files by filename or content match within a sandboxed directory.",
        args_model=SearchFilesArgs,
        handler=handler,
    )
