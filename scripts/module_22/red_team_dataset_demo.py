"""Lab 1 - build a red-team prompt dataset. Loads the real, committed,
hand-labeled `datasets/red_team/red_team_prompts.jsonl` and reports its
real composition (malicious/benign split, category counts, threat-surface
coverage) - the numbers below come from the actual file, not from how it
was designed to look.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "red_team" / "red_team_prompts.jsonl"


def load_examples() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_lab() -> dict:
    examples = load_examples()
    malicious_count = sum(1 for e in examples if e["is_malicious"])
    benign_count = len(examples) - malicious_count

    return {
        "total": len(examples),
        "malicious_count": malicious_count,
        "benign_count": benign_count,
        "category_counts": dict(Counter(e["category"] for e in examples)),
        "surface_counts": dict(Counter(e["surface"] for e in examples)),
    }


def result_to_markdown(result: dict) -> str:
    lines = [
        "# Lab 1 - red-team prompt dataset",
        "",
        f"- Total examples: {result['total']}",
        f"- Malicious: {result['malicious_count']}, Benign: {result['benign_count']}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(result["category_counts"].items()):
        lines.append(f"| {category} | {count} |")
    lines.append("")
    lines.append("| Threat surface | Count |")
    lines.append("|---|---:|")
    for surface, count in sorted(result["surface_counts"].items()):
        lines.append(f"| {surface} | {count} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
