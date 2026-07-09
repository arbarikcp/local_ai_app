"""search_repo tool (curriculum's required tools list) - like Module 14's
`search_files`, but returns matching line numbers too (grep-like
precision), tailored for repo/code search rather than generic file search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.sandbox import resolve_within_sandbox


class SearchRepoArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    root_path: str = Field(default=".")
    max_results: int = Field(default=20, ge=1, le=100)


@dataclass(frozen=True)
class RepoMatch:
    path: str
    line_number: int
    line_text: str


def search_repo(allowed_base: Path, query: str, root_path: str = ".", max_results: int = 20) -> list[RepoMatch]:
    root = resolve_within_sandbox(allowed_base, root_path)
    base = allowed_base.resolve()
    matches: list[RepoMatch] = []

    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, start=1):
            if query.lower() in line.lower():
                matches.append(RepoMatch(path=str(path.resolve().relative_to(base)), line_number=i, line_text=line.strip()))
            if len(matches) >= max_results:
                return matches
    return matches


def make_search_repo_tool(allowed_base: Path) -> Tool:
    async def handler(args: SearchRepoArgs) -> list[dict]:
        return [asdict(m) for m in search_repo(allowed_base, args.query, args.root_path, args.max_results)]

    return Tool(
        name="search_repo",
        description="Search Python files in the repo for a query string, returning matched line numbers.",
        args_model=SearchRepoArgs,
        handler=handler,
    )
