"""Context assembly (ARCHITECTURE.md "Context building") — composes
Module 17's `list_symbols`/`search_repo`/`read_file_lines` into one bundle
per intent. Nothing in the repo does this composition (confirmed by
survey: `coding_assistant.py` assembles context ad hoc, inline, for one
fixed pipeline shape only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from local_ai_agents.tools.list_symbols import Symbol, list_symbols
from local_ai_agents.tools.read_file import read_file_lines
from local_ai_agents.tools.search_repo import RepoMatch, search_repo

from eng_intent_classifier import IntentType

_EXCERPT_LINES_AFTER_SYMBOL = 20


class SymbolNotFoundError(Exception):
    def __init__(self, symbol_name: str) -> None:
        super().__init__(f"no symbol named {symbol_name!r} found in the repo")
        self.symbol_name = symbol_name


@dataclass(frozen=True)
class ContextBundle:
    intent: IntentType
    repo_symbols: dict[str, list[Symbol]] = field(default_factory=dict)
    search_results: list[RepoMatch] = field(default_factory=list)
    file_excerpt: str | None = None


def _iter_python_files(allowed_base: Path) -> list[str]:
    return [str(p.relative_to(allowed_base)) for p in sorted(allowed_base.rglob("*.py"))]


def _find_symbol(allowed_base: Path, symbol_name: str) -> tuple[str, Symbol] | None:
    for relative_path in _iter_python_files(allowed_base):
        try:
            symbols = list_symbols(allowed_base, relative_path)
        except SyntaxError:
            continue
        for symbol in symbols:
            if symbol.name == symbol_name:
                return relative_path, symbol
    return None


def build_context(
    allowed_base: Path,
    intent: IntentType,
    *,
    query: str | None = None,
    target_file: str | None = None,
    symbol_name: str | None = None,
) -> ContextBundle:
    if intent == IntentType.EXPLAIN_REPO:
        repo_symbols = {}
        for relative_path in _iter_python_files(allowed_base):
            try:
                repo_symbols[relative_path] = list_symbols(allowed_base, relative_path)
            except SyntaxError:
                continue
        return ContextBundle(intent=intent, repo_symbols=repo_symbols)

    if intent == IntentType.SEARCH_CODE:
        if not query:
            raise ValueError("query is required for the search_code intent")
        return ContextBundle(intent=intent, search_results=search_repo(allowed_base, query))

    if intent in (IntentType.EXPLAIN_SYMBOL, IntentType.PROPOSE_PATCH, IntentType.GENERATE_TESTS, IntentType.SUGGEST_REFACTOR):
        if symbol_name:
            found = _find_symbol(allowed_base, symbol_name)
            if found is None:
                raise SymbolNotFoundError(symbol_name)
            relative_path, symbol = found
            excerpt = read_file_lines(allowed_base, relative_path, symbol.line, symbol.line + _EXCERPT_LINES_AFTER_SYMBOL)
            return ContextBundle(intent=intent, repo_symbols={relative_path: [symbol]}, file_excerpt=excerpt)
        if target_file:
            symbols = list_symbols(allowed_base, target_file)
            excerpt = read_file_lines(allowed_base, target_file)
            return ContextBundle(intent=intent, repo_symbols={target_file: symbols}, file_excerpt=excerpt)
        raise ValueError("symbol_name or target_file is required for this intent")

    # RUN_TESTS needs no repo context - it just executes the test suite.
    return ContextBundle(intent=intent)
