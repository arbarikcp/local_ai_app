"""Write-file tool (Lab 5) - the one dangerous tool this module
implements. Path containment (`sandbox.py`, same mechanism as
`file_search.py`) *plus* `dangerous=True`, so `ToolExecutor` refuses to
run it without a real `ApprovalGate` approval (theory doc §9-10).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.sandbox import resolve_within_sandbox


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=100_000)


def write_file(allowed_base: Path, relative_path: str, content: str) -> str:
    target = resolve_within_sandbox(allowed_base, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.relative_to(allowed_base.resolve()))


def make_write_file_tool(allowed_base: Path) -> Tool:
    async def handler(args: WriteFileArgs) -> str:
        return write_file(allowed_base, args.path, args.content)

    return Tool(
        name="write_file",
        description="Write content to a file within the sandboxed directory. Requires human approval.",
        args_model=WriteFileArgs,
        handler=handler,
        dangerous=True,
    )
