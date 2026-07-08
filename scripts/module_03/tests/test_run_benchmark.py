import json
from pathlib import Path

import pytest

import run_benchmark as sut

GOLDEN_SETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "evals" / "golden_sets"


@pytest.mark.parametrize(
    "filename",
    [
        "summarization.jsonl",
        "extraction.jsonl",
        "classification.jsonl",
        "code.jsonl",
        "rag.jsonl",
        "tool_calling.jsonl",
    ],
)
def test_golden_set_files_exist_and_parse(filename):
    path = GOLDEN_SETS_DIR / filename
    assert path.exists(), f"missing golden set: {path}"
    dataset = sut.load_dataset(path)
    assert len(dataset) >= 4
    for record in dataset:
        assert "id" in record
        assert "task" in record
        assert "scorer" in record


@pytest.mark.parametrize(
    "filename",
    [
        "summarization.jsonl",
        "extraction.jsonl",
        "classification.jsonl",
        "code.jsonl",
        "rag.jsonl",
        "tool_calling.jsonl",
    ],
)
def test_prompt_for_record_builds_a_nonempty_prompt_for_every_record(filename):
    dataset = sut.load_dataset(GOLDEN_SETS_DIR / filename)
    for record in dataset:
        prompt = sut.prompt_for_record(record)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_prompt_for_record_raises_on_unknown_task():
    with pytest.raises(ValueError):
        sut.prompt_for_record({"task": "not-a-real-task"})


def test_score_record_contains_all():
    record = {"scorer": "contains_all", "required_facts": ["Paris", "1889"]}
    assert sut.score_record(record, "Built in 1889 in Paris.") == 1.0
    assert sut.score_record(record, "Built somewhere.") == 0.0


def test_score_record_json_validity_and_keys():
    record = {"scorer": "json_validity_and_keys", "schema_keys": ["name", "age"]}
    assert sut.score_record(record, '{"name": "A", "age": 1}') == 1.0
    assert sut.score_record(record, "not json") == 0.0


def test_score_record_normalized_exact_match():
    record = {"scorer": "normalized_exact_match", "reference_label": "billing"}
    assert sut.score_record(record, "Billing.") == 1.0
    assert sut.score_record(record, "shipping") == 0.0


def test_score_record_citation_validity():
    record = {
        "scorer": "citation_validity",
        "context": [{"doc_id": "doc1", "text": "x"}, {"doc_id": "doc2", "text": "y"}],
    }
    assert sut.score_record(record, "Fact [doc1] and fact [doc2].") == 1.0
    assert sut.score_record(record, "Fact [doc1] and fabricated [doc9].") == pytest.approx(0.5)


def test_score_record_grounded_refusal():
    record = {"scorer": "grounded_refusal", "refusal_phrase": "I don't know based on the provided documents"}
    assert sut.score_record(record, "I don't know based on the provided documents.") == 1.0
    assert sut.score_record(record, "The answer is 42.") == 0.0


def test_score_record_unknown_scorer_raises():
    with pytest.raises(ValueError):
        sut.score_record({"scorer": "not-a-real-scorer"}, "prediction")


class TestToolCallScoring:
    def test_correct_tool_and_full_argument_match_scores_one(self):
        record = {
            "scorer": "tool_call_validity",
            "expected_tool": "get_weather",
            "expected_arguments": {"city": "Tokyo", "unit": "celsius"},
        }
        prediction = json.dumps({"function": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}})
        assert sut.score_record(record, prediction) == 1.0

    def test_correct_tool_partial_argument_match_scores_partial(self):
        record = {
            "scorer": "tool_call_validity",
            "expected_tool": "get_weather",
            "expected_arguments": {"city": "Tokyo", "unit": "celsius"},
        }
        prediction = json.dumps({"function": "get_weather", "arguments": {"city": "Tokyo", "unit": "fahrenheit"}})
        assert sut.score_record(record, prediction) == pytest.approx(0.5)

    def test_wrong_tool_scores_zero(self):
        record = {"scorer": "tool_call_validity", "expected_tool": "get_weather", "expected_arguments": {}}
        prediction = json.dumps({"function": "get_time", "arguments": {}})
        assert sut.score_record(record, prediction) == 0.0

    def test_negative_case_no_tool_call_scores_one(self):
        record = {"scorer": "tool_call_validity", "expected_tool": None, "expected_arguments": None}
        assert sut.score_record(record, "NO_TOOL") == 1.0

    def test_negative_case_hallucinated_tool_call_scores_zero(self):
        record = {"scorer": "tool_call_validity", "expected_tool": None, "expected_arguments": None}
        prediction = json.dumps({"function": "get_weather", "arguments": {"city": "Nowhere"}})
        assert sut.score_record(record, prediction) == 0.0


def test_run_task_benchmark_uses_injected_generate_fn_and_scores_each_record():
    dataset = [
        {
            "id": "x1",
            "task": "classification",
            "text": "billing issue",
            "labels": ["billing", "shipping"],
            "reference_label": "billing",
            "scorer": "normalized_exact_match",
        }
    ]

    def fake_generate(model: str, prompt: str) -> str:
        return "billing"

    result = sut.run_task_benchmark("fake-model", dataset, fake_generate)
    assert result.task_name == "classification"
    assert result.model == "fake-model"
    assert len(result.record_results) == 1
    assert result.record_results[0].score == 1.0
    assert result.mean_score == 1.0


def test_run_task_benchmark_empty_dataset_has_zero_mean_score():
    result = sut.run_task_benchmark("fake-model", [], lambda m, p: "")
    assert result.mean_score == 0.0
    assert result.task_name == "unknown"


def test_run_full_benchmark_runs_every_model_against_every_dataset(tmp_path):
    ds_path = tmp_path / "cls.jsonl"
    ds_path.write_text(
        json.dumps(
            {
                "id": "c1",
                "task": "classification",
                "text": "t",
                "labels": ["a", "b"],
                "reference_label": "a",
                "scorer": "normalized_exact_match",
            }
        )
        + "\n"
    )

    def fake_generate(model: str, prompt: str) -> str:
        return "a" if model == "good-model" else "b"

    results = sut.run_full_benchmark(["good-model", "bad-model"], {"cls": ds_path}, fake_generate)
    assert set(results.keys()) == {"good-model", "bad-model"}
    assert results["good-model"][0].mean_score == 1.0
    assert results["bad-model"][0].mean_score == 0.0


def test_scorecard_quality_table_renders_task_rows():
    task_result = sut.TaskResult(
        task_name="classification",
        model="m",
        record_results=[sut.RecordResult(record_id="1", prediction="a", score=1.0)],
    )
    table = sut.scorecard_quality_table([task_result])
    assert "classification" in table
    assert "1.00" in table


def test_comparison_table_renders_all_models_and_tasks():
    task_result_good = sut.TaskResult(
        task_name="classification", model="good", record_results=[sut.RecordResult("1", "a", 1.0)]
    )
    task_result_bad = sut.TaskResult(
        task_name="classification", model="bad", record_results=[sut.RecordResult("1", "b", 0.0)]
    )
    table = sut.comparison_table({"good": [task_result_good], "bad": [task_result_bad]})
    assert "good" in table
    assert "bad" in table
    assert "classification" in table


def test_comparison_table_empty_results_returns_empty_string():
    assert sut.comparison_table({}) == ""
