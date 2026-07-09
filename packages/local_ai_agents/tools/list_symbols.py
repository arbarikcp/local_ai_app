"""list_symbols tool (curriculum's required tools list; theory doc §3-4,
AST-aware parsing / symbol search) - real Python `ast` module parsing,
not a regex-based guess at function/class boundaries.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.sandbox import resolve_within_sandbox


class ListSymbolsArgs(BaseModel):
    path: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # "function" | "async_function" | "class"
    line: int


def list_symbols(allowed_base: Path, relative_path: str) -> list[Symbol]:
    target = resolve_within_sandbox(allowed_base, relative_path)
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))

    symbols: list[Symbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            symbols.append(Symbol(name=node.name, kind="async_function", line=node.lineno))
        elif isinstance(node, ast.FunctionDef):
            symbols.append(Symbol(name=node.name, kind="function", line=node.lineno))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(name=node.name, kind="class", line=node.lineno))
    return sorted(symbols, key=lambda s: s.line)


def make_list_symbols_tool(allowed_base: Path) -> Tool:
    async def handler(args: ListSymbolsArgs) -> list[dict]:
        return [asdict(s) for s in list_symbols(allowed_base, args.path)]

    return Tool(
        name="list_symbols",
        description="List function/class symbols defined in a Python file, via real AST parsing.",
        args_model=ListSymbolsArgs,
        handler=handler,
    )
