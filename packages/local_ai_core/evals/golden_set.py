"""Golden RAG dataset loading (theory doc §1, "RAG evaluation dataset") -
curriculum's exact schema, loaded from JSONL (one JSON object per line, the
same format Module 3/8's golden sets under `evals/golden_sets/` already use
elsewhere in this repo).

`expected_source_ids` holds **document** ids (`"password_reset"`), not
full `chunk_id`s (`"password_reset::0"`) - the same doc-level-relevance
choice Module 11's `qa_eval.py` made, since chunk boundaries shift with
chunk size (Module 12's chunking-strategy comparisons) while document
identity doesn't. `retrieval_metrics.py`'s callers are expected to reduce
retrieved `chunk_id`s to their `doc_id` prefix before comparing against
`expected_source_ids`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class GoldenCase:
    question_id: str
    question: str
    expected_answer: str
    expected_source_ids: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    difficulty: Difficulty = "medium"
    category: str = "general"

    @property
    def requires_refusal(self) -> bool:
        """No expected source documents means no chunk in the corpus should
        answer this - the case exists specifically to test abstention
        (theory doc §12, "Refusal behavior").
        """
        return len(self.expected_source_ids) == 0


def load_golden_set(path: Path) -> list[GoldenCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                GoldenCase(
                    question_id=row["question_id"],
                    question=row["question"],
                    expected_answer=row["expected_answer"],
                    expected_source_ids=row.get("expected_source_ids", []),
                    must_contain=row.get("must_contain", []),
                    must_not_contain=row.get("must_not_contain", []),
                    difficulty=row.get("difficulty", "medium"),
                    category=row.get("category", "general"),
                )
            )
    return cases
