"""Labs 1-2 - index a small Python repo (real AST-based symbol listing
across every `.py` file) and answer an architecture question ("what
functions exist in calculator.py?") from that real data, plus a real
lexical repo search. Runs for real, no live model needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_agents.tools.list_symbols import list_symbols  # noqa: E402
from local_ai_agents.tools.search_repo import search_repo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_REPO = REPO_ROOT / "datasets" / "code_repos" / "mini_calculator"


def build_repo_map(repo_dir: Path) -> dict[str, list[dict]]:
    repo_map = {}
    for path in sorted(repo_dir.rglob("*.py")):
        relative = str(path.relative_to(repo_dir))
        symbols = list_symbols(repo_dir, relative)
        repo_map[relative] = [{"name": s.name, "kind": s.kind, "line": s.line} for s in symbols]
    return repo_map


async def run_lab() -> dict:
    repo_map = build_repo_map(SAMPLE_REPO)

    # Lab 2: an "architecture question" answered from real AST data, not a guess.
    calculator_functions = [s["name"] for s in repo_map["calculator.py"] if s["kind"] == "function"]

    # A real lexical search across the repo.
    matches = search_repo(SAMPLE_REPO, "ValueError")

    return {
        "files_indexed": list(repo_map.keys()),
        "repo_map": repo_map,
        "calculator_functions": calculator_functions,
        "search_matches": [f"{m.path}:{m.line_number}" for m in matches],
    }


def result_to_markdown(result: dict) -> str:
    lines = ["# Labs 1-2 - repo index and architecture questions\n"]
    lines.append(f"- Files indexed: {result['files_indexed']}\n")
    lines.append("## Repo map (real AST symbols)\n")
    for path, symbols in result["repo_map"].items():
        lines.append(f"- {path}:")
        for s in symbols:
            lines.append(f"  - {s['kind']} {s['name']} (line {s['line']})")
    lines.append(f"\n## \"What functions exist in calculator.py?\"\n{result['calculator_functions']}\n")
    lines.append(f"\n## search_repo('ValueError')\n{result['search_matches']}\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
