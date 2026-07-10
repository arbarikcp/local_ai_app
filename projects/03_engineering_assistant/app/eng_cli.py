"""Lab 1 - a real `typer.Typer()` CLI (ARCHITECTURE.md "CLI"), curriculum's
own "CLI: best for developer tools and labs" framing. Every command routes
through `EngAppContext.tool_executor`, so every capability is audit-logged
- not just `apply_patch`/`run_tests`, the only two Module 17's
`WorkflowExecutor` gated at the node level.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))
sys.path.insert(0, str(_PROJECT_ROOT / "prompts"))

import typer  # noqa: E402

from local_ai_agents.policies.approval import AutoApprovalGate  # noqa: E402
from local_ai_agents.tools.base import ToolCallProposal  # noqa: E402
from local_ai_core.deployment.config import load_config  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

from eng_prompts import (  # noqa: E402
    build_explain_symbol_prompt,
    build_generate_tests_prompt,
    build_suggest_refactor_prompt,
)
from eng_service import EngAppContext, build_eng_context  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"
DEFAULT_REPO_DIR = _PROJECT_ROOT / "demo_repo"
_EXCERPT_LINES_AFTER_SYMBOL = 20

app = typer.Typer(help="Local engineering assistant CLI (Project 3).")


def _build_context(repo_dir: str, config_path: str, *, approve: bool) -> EngAppContext:
    config = load_config(config_path)
    approval_gate = AutoApprovalGate() if approve else None
    return build_eng_context(
        config,
        model_catalog_path=DEFAULT_CATALOG_PATH,
        repo_dir=Path(repo_dir),
        approval_gate=approval_gate,
    )


async def _call(ctx: EngAppContext, tool_name: str, arguments: dict):
    return await ctx.tool_executor.execute(
        ToolCallProposal(tool_name=tool_name, raw_arguments=arguments), trace_id=str(uuid.uuid4())
    )


def _list_python_files(repo_dir: Path) -> list[str]:
    return [str(p.relative_to(repo_dir)) for p in sorted(repo_dir.rglob("*.py"))]


async def _locate_symbol_excerpt(ctx: EngAppContext, symbol_name: str) -> tuple[str, str] | None:
    for relative_path in _list_python_files(ctx.repo_dir):
        list_result = await _call(ctx, "list_symbols", {"path": relative_path})
        if not list_result.success:
            continue
        match = next((s for s in list_result.data if s["name"] == symbol_name), None)
        if match is None:
            continue
        read_result = await _call(
            ctx, "read_file", {"path": relative_path, "start_line": match["line"], "end_line": match["line"] + _EXCERPT_LINES_AFTER_SYMBOL}
        )
        if read_result.success:
            return relative_path, read_result.data
    return None


@app.command()
def explain_repo(
    repo_dir: str = str(DEFAULT_REPO_DIR), config_path: str = str(DEFAULT_CONFIG_PATH)
) -> None:
    """List every function/class symbol across the repo, real AST parsing per file."""
    ctx = _build_context(repo_dir, config_path, approve=False)

    async def _run() -> dict:
        results = {}
        for relative_path in _list_python_files(ctx.repo_dir):
            result = await _call(ctx, "list_symbols", {"path": relative_path})
            if result.success:
                results[relative_path] = result.data
        return results

    for relative_path, symbols in asyncio.run(_run()).items():
        typer.echo(relative_path)
        for symbol in symbols:
            typer.echo(f"  {symbol['kind']} {symbol['name']} (line {symbol['line']})")


@app.command()
def search(query: str, repo_dir: str = str(DEFAULT_REPO_DIR), config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """Search the repo for a query string, real line numbers."""
    ctx = _build_context(repo_dir, config_path, approve=False)
    result = asyncio.run(_call(ctx, "search_repo", {"query": query}))
    if not result.success:
        typer.echo(f"Search failed: {result.error_message}")
        raise typer.Exit(code=1)
    for match in result.data:
        typer.echo(f"{match['path']}:{match['line_number']}: {match['line_text']}")


@app.command()
def explain_symbol(
    symbol_name: str, repo_dir: str = str(DEFAULT_REPO_DIR), config_path: str = str(DEFAULT_CONFIG_PATH)
) -> None:
    """Explain a function/class, real AST location + a real LLM call."""
    ctx = _build_context(repo_dir, config_path, approve=False)

    async def _run() -> str | None:
        found = await _locate_symbol_excerpt(ctx, symbol_name)
        if found is None:
            return None
        _, excerpt = found
        prompt = build_explain_symbol_prompt(symbol_name, excerpt)
        response = await ctx.base.runtime.generate(LLMRequest(model=ctx.base.config.models.default_code, prompt=prompt))
        return response.text

    text = asyncio.run(_run())
    if text is None:
        typer.echo(f"No symbol named {symbol_name!r} found.")
        raise typer.Exit(code=1)
    typer.echo(text)


@app.command()
def generate_tests(
    symbol_name: str, repo_dir: str = str(DEFAULT_REPO_DIR), config_path: str = str(DEFAULT_CONFIG_PATH)
) -> None:
    """Generate a pytest test function for a symbol, real AST location + a real LLM call."""
    ctx = _build_context(repo_dir, config_path, approve=False)

    async def _run() -> str | None:
        found = await _locate_symbol_excerpt(ctx, symbol_name)
        if found is None:
            return None
        _, excerpt = found
        prompt = build_generate_tests_prompt(symbol_name, excerpt)
        response = await ctx.base.runtime.generate(LLMRequest(model=ctx.base.config.models.default_code, prompt=prompt))
        return response.text

    text = asyncio.run(_run())
    if text is None:
        typer.echo(f"No symbol named {symbol_name!r} found.")
        raise typer.Exit(code=1)
    typer.echo(text)


@app.command()
def suggest_refactor(
    target_file: str, repo_dir: str = str(DEFAULT_REPO_DIR), config_path: str = str(DEFAULT_CONFIG_PATH)
) -> None:
    """Suggest a refactor for a whole file, real file read + a real LLM call."""
    ctx = _build_context(repo_dir, config_path, approve=False)

    async def _run() -> str:
        read_result = await _call(ctx, "read_file", {"path": target_file})
        prompt = build_suggest_refactor_prompt(read_result.data)
        response = await ctx.base.runtime.generate(LLMRequest(model=ctx.base.config.models.default_code, prompt=prompt))
        return response.text

    typer.echo(asyncio.run(_run()))


@app.command()
def propose_patch(
    instruction: str,
    target_file: str,
    repo_dir: str = str(DEFAULT_REPO_DIR),
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> None:
    """Propose a unified-diff patch for a file, real LLM call."""
    ctx = _build_context(repo_dir, config_path, approve=False)

    async def _run():
        read_result = await _call(ctx, "read_file", {"path": target_file})
        return await _call(
            ctx, "propose_patch", {"instruction": instruction, "file_contents": {target_file: read_result.data}}
        )

    result = asyncio.run(_run())
    if not result.success:
        typer.echo(f"Failed: {result.error_message}")
        raise typer.Exit(code=1)
    typer.echo(result.data)


@app.command()
def apply_patch(
    patch_file: str,
    expected_file_path: str,
    repo_dir: str = str(DEFAULT_REPO_DIR),
    config_path: str = str(DEFAULT_CONFIG_PATH),
    approve: bool = typer.Option(False, "--approve", help="Approve this dangerous operation."),
) -> None:
    """Apply a validated unified-diff patch. Requires --approve."""
    ctx = _build_context(repo_dir, config_path, approve=approve)
    patch_text = Path(patch_file).read_text(encoding="utf-8")
    result = asyncio.run(
        _call(ctx, "apply_patch", {"patch_text": patch_text, "expected_file_path": expected_file_path})
    )
    if not result.success:
        typer.echo(f"Failed: {result.error_message}")
        raise typer.Exit(code=1)
    typer.echo(f"Applied patch to {result.data}")


@app.command()
def run_tests(
    test_path: str = "tests",
    repo_dir: str = str(DEFAULT_REPO_DIR),
    config_path: str = str(DEFAULT_CONFIG_PATH),
    approve: bool = typer.Option(False, "--approve", help="Approve this dangerous operation."),
) -> None:
    """Run the repo's pytest suite. Requires --approve."""
    ctx = _build_context(repo_dir, config_path, approve=approve)
    result = asyncio.run(_call(ctx, "run_tests", {"test_path": test_path}))
    if not result.success:
        typer.echo(f"Failed: {result.error_message}")
        raise typer.Exit(code=1)
    typer.echo(result.data["stdout"])
    if not result.data["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
