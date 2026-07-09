"""Labs 6-7 - run the local guard classifier against the red-team set and
measure catch rate, false positives, false negatives, and latency. Real
numbers from a real classifier run against every example in the committed
dataset - not a hand-picked subset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.security.guard_eval import LabeledExample, evaluate_guard_classifier  # noqa: E402
from local_ai_core.security.guard_pipeline import RuleBasedGuardClassifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "red_team" / "red_team_prompts.jsonl"


def load_examples() -> list[LabeledExample]:
    examples = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                LabeledExample(text=record["text"], is_malicious=record["is_malicious"], label=record["category"])
            )
    return examples


def run_lab() -> dict:
    examples = load_examples()
    classifier = RuleBasedGuardClassifier()
    report = evaluate_guard_classifier(classifier, examples)

    false_negative_examples = [
        e.text
        for e in examples
        if e.is_malicious and classifier.classify(e.text).verdict.value == "allow"
    ]
    false_positive_examples = [
        e.text
        for e in examples
        if not e.is_malicious and classifier.classify(e.text).verdict.value != "allow"
    ]

    return {
        "total": report.total,
        "true_positives": report.true_positives,
        "false_positives": report.false_positives,
        "true_negatives": report.true_negatives,
        "false_negatives": report.false_negatives,
        "catch_rate": report.catch_rate,
        "false_positive_rate": report.false_positive_rate,
        "mean_latency_ms": report.mean_latency_ms,
        "false_negative_examples": false_negative_examples,
        "false_positive_examples": false_positive_examples,
    }


def result_to_markdown(result: dict) -> str:
    lines = [
        "# Labs 6-7 - guard classifier evaluation against the red-team set",
        "",
        f"- Total examples: {result['total']}",
        f"- TP={result['true_positives']} FP={result['false_positives']} "
        f"TN={result['true_negatives']} FN={result['false_negatives']}",
        f"- Catch rate (recall): {result['catch_rate']:.1%}",
        f"- False positive rate: {result['false_positive_rate']:.1%}",
        f"- Mean latency: {result['mean_latency_ms']:.4f}ms",
        "",
    ]
    if result["false_negative_examples"]:
        lines.append("## Missed (false negatives)")
        for text in result["false_negative_examples"]:
            lines.append(f"- \"{text}\"")
        lines.append("")
    if result["false_positive_examples"]:
        lines.append("## Wrongly flagged (false positives)")
        for text in result["false_positive_examples"]:
            lines.append(f"- \"{text}\"")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
