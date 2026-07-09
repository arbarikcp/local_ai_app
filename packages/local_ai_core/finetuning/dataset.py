"""Instruction-tuning dataset schema, cleaning, and splitting (theory doc
§5-7). Real, deterministic Python over `TrainingExample` records - no model
dependency, unlike `mlx_lora.py`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingExample:
    instruction: str
    input: str
    output: str

    def dedup_key(self) -> tuple[str, str]:
        return (self.instruction, self.input)


@dataclass(frozen=True)
class DropRecord:
    example: TrainingExample
    reason: str


@dataclass(frozen=True)
class CleaningResult:
    kept: list[TrainingExample]
    dropped: list[DropRecord]


@dataclass(frozen=True)
class DatasetSplit:
    train: list[TrainingExample]
    validation: list[TrainingExample]
    test: list[TrainingExample]


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                TrainingExample(
                    instruction=record["instruction"],
                    input=record["input"],
                    output=record["output"],
                )
            )
    return examples


def clean_dataset(
    examples: list[TrainingExample],
    *,
    min_output_chars: int = 1,
    max_input_chars: int = 2000,
) -> CleaningResult:
    """Real deduplication (exact-match on instruction+input) and real length
    filtering, each returning *why* an example was dropped rather than just
    a smaller list.
    """
    kept: list[TrainingExample] = []
    dropped: list[DropRecord] = []
    seen: set[tuple[str, str]] = set()

    for example in examples:
        key = example.dedup_key()
        if key in seen:
            dropped.append(DropRecord(example=example, reason="duplicate instruction+input"))
            continue
        if len(example.output) < min_output_chars:
            dropped.append(DropRecord(example=example, reason="output shorter than min_output_chars"))
            continue
        if len(example.input) > max_input_chars:
            dropped.append(DropRecord(example=example, reason="input longer than max_input_chars"))
            continue
        seen.add(key)
        kept.append(example)

    return CleaningResult(kept=kept, dropped=dropped)


def split_dataset(
    examples: list[TrainingExample],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """A real, seeded random split at configurable ratios. The remainder
    after train and validation goes to test, so the three ratios always sum
    to the full dataset regardless of rounding.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_validation = round(n * validation_ratio)

    return DatasetSplit(
        train=shuffled[:n_train],
        validation=shuffled[n_train : n_train + n_validation],
        test=shuffled[n_train + n_validation :],
    )


def detect_leakage(split: DatasetSplit) -> list[tuple[str, str]]:
    """A real check that no example's (instruction, input) pair appears in
    more than one split - returns the leaked keys, empty when the split is
    clean.
    """
    seen_in: dict[tuple[str, str], set[str]] = {}
    for split_name, split_examples in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for example in split_examples:
            key = example.dedup_key()
            seen_in.setdefault(key, set()).add(split_name)

    return [key for key, splits in seen_in.items() if len(splits) > 1]
