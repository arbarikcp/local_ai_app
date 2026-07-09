"""Lab 1 - create a labeled dataset, clean it, split it, and prove the
split has no leakage. Runs entirely for real: the committed
`ticket_classification.jsonl` (40 hand-labeled Nimbus support tickets), a
deliberately-duplicated copy to exercise cleaning, and a real seeded
train/validation/test split checked with `detect_leakage()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.finetuning.dataset import (  # noqa: E402
    clean_dataset,
    detect_leakage,
    load_jsonl,
    split_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "finetuning" / "ticket_classification.jsonl"


def run_lab() -> dict:
    examples = load_jsonl(DATASET_PATH)

    # Exercise real deduplication by re-adding the first example as a
    # duplicate before cleaning.
    examples_with_duplicate = [*examples, examples[0]]
    cleaning_result = clean_dataset(examples_with_duplicate)

    split = split_dataset(cleaning_result.kept, train_ratio=0.8, validation_ratio=0.1, seed=42)
    leaks = detect_leakage(split)

    return {
        "raw_example_count": len(examples_with_duplicate),
        "cleaned_example_count": len(cleaning_result.kept),
        "dropped_count": len(cleaning_result.dropped),
        "dropped_reasons": [record.reason for record in cleaning_result.dropped],
        "train_count": len(split.train),
        "validation_count": len(split.validation),
        "test_count": len(split.test),
        "leak_count": len(leaks),
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 1 - dataset creation, cleaning, splitting\n\n"
        f"- Raw examples (with one deliberate duplicate): {result['raw_example_count']}\n"
        f"- Cleaned examples: {result['cleaned_example_count']} "
        f"(dropped {result['dropped_count']}: {result['dropped_reasons']})\n"
        f"- Split: train={result['train_count']}, validation={result['validation_count']}, "
        f"test={result['test_count']}\n"
        f"- Leakage check: {result['leak_count']} leaked (instruction, input) pairs\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
