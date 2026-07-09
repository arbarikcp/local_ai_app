import json

from local_ai_core.evals.golden_set import GoldenCase, load_golden_set


def write_jsonl(tmp_path, rows: list[dict]):
    path = tmp_path / "golden.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class TestLoadGoldenSet:
    def test_loads_every_row(self, tmp_path):
        rows = [
            {"question_id": "q1", "question": "Q1?", "expected_answer": "A1", "expected_source_ids": ["d1"]},
            {"question_id": "q2", "question": "Q2?", "expected_answer": "A2", "expected_source_ids": ["d2"]},
        ]
        path = write_jsonl(tmp_path, rows)
        cases = load_golden_set(path)
        assert len(cases) == 2
        assert cases[0].question_id == "q1"

    def test_defaults_are_applied_for_optional_fields(self, tmp_path):
        rows = [{"question_id": "q1", "question": "Q1?", "expected_answer": "A1"}]
        path = write_jsonl(tmp_path, rows)
        cases = load_golden_set(path)
        assert cases[0].expected_source_ids == []
        assert cases[0].must_contain == []
        assert cases[0].difficulty == "medium"
        assert cases[0].category == "general"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"question_id": "q1", "question": "Q1?", "expected_answer": "A1"}\n\n', encoding="utf-8"
        )
        cases = load_golden_set(path)
        assert len(cases) == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_golden_set(path) == []


class TestGoldenCaseRequiresRefusal:
    def test_true_when_no_expected_source_ids(self):
        case = GoldenCase(question_id="q1", question="Q?", expected_answer="I don't know")
        assert case.requires_refusal is True

    def test_false_when_expected_source_ids_present(self):
        case = GoldenCase(question_id="q1", question="Q?", expected_answer="A", expected_source_ids=["d1"])
        assert case.requires_refusal is False
