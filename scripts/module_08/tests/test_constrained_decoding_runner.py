import pytest

import constrained_decoding_runner as sut
from local_ai_core.runtimes.fake import FakeRuntime


class TestLoadMatchingGoldenCases:
    def test_returns_only_records_matching_the_schema_keys(self):
        cases = sut.load_matching_golden_cases()
        assert len(cases) == 2  # ext-001 and ext-005, per the golden set's current content
        for case in cases:
            assert set(case["schema_keys"]) == sut.PERSON_SCHEMA_KEYS

    def test_every_case_has_a_reference_dict(self):
        cases = sut.load_matching_golden_cases()
        for case in cases:
            assert "reference" in case


class TestFieldAccuracy:
    def test_perfect_match_is_full_accuracy(self):
        acc = sut.field_accuracy(
            {"name": "Maria", "age": 29, "city": "Austin"}, {"name": "Maria", "age": 29, "city": "Austin"}
        )
        assert acc == 1.0

    def test_partial_match_is_partial_accuracy(self):
        acc = sut.field_accuracy({"name": "Maria", "age": 30, "city": "Austin"}, {"name": "Maria", "age": 29, "city": "Austin"})
        assert acc == pytest.approx(2 / 3)

    def test_correctly_predicted_null_counts_as_correct(self):
        # ext-005: age is genuinely unknown; correctly returning null is a
        # correct prediction, not a miss.
        acc = sut.field_accuracy({"name": "Priya", "age": None, "city": "Chicago"}, {"name": "Priya", "age": None, "city": "Chicago"})
        assert acc == 1.0

    def test_fabricated_value_for_a_null_reference_is_wrong(self):
        acc = sut.field_accuracy({"name": "Priya", "age": 40, "city": "Chicago"}, {"name": "Priya", "age": None, "city": "Chicago"})
        assert acc == pytest.approx(2 / 3)

    def test_empty_reference_returns_zero(self):
        assert sut.field_accuracy({"a": 1}, {}) == 0.0


class TestRunMode:
    async def test_perfect_runtime_gives_zero_invalid_rate_and_full_accuracy(self):
        cases = [{"id": "c1", "text": "Maria is 29 in Austin.", "reference": {"name": "Maria", "age": 29, "city": "Austin"}}]
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        result = await sut.run_mode("json_schema", runtime, cases, "m")
        assert result.invalid_json_rate == 0.0
        assert result.field_accuracy == 1.0
        assert result.used_constrained_decoding_rate == 1.0

    async def test_broken_runtime_gives_full_invalid_rate(self):
        cases = [{"id": "c1", "text": "text", "reference": {"name": "X", "age": 1, "city": "Y"}}]
        runtime = FakeRuntime(default_response="not json at all", simulated_latency_ms=0.0)
        result = await sut.run_mode("text", runtime, cases, "m")
        assert result.invalid_json_rate == 1.0
        assert result.field_accuracy == 0.0

    async def test_text_mode_never_reports_constrained_decoding_used(self):
        cases = [{"id": "c1", "text": "text", "reference": {"name": "X", "age": 1, "city": "Y"}}]
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        result = await sut.run_mode("text", runtime, cases, "m")
        assert result.used_constrained_decoding_rate == 0.0


class TestRunLab:
    async def test_produces_one_result_per_mode(self):
        cases = [{"id": "c1", "text": "text", "reference": {"name": "X", "age": 1, "city": "Y"}}]
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        results = await sut.run_lab(runtime, "m", cases)
        assert [r.mode for r in results] == sut.MODES

    async def test_uses_real_golden_cases_when_none_given(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        results = await sut.run_lab(runtime, "m")
        assert len(results) == len(sut.MODES)


class TestResultsToMarkdownTable:
    def test_renders_all_modes(self):
        results = [
            sut.ModeResult(mode="text", invalid_json_rate=0.5, field_accuracy=0.5, p95_latency_seconds=0.01, used_constrained_decoding_rate=0.0),
            sut.ModeResult(mode="json_schema", invalid_json_rate=0.0, field_accuracy=1.0, p95_latency_seconds=0.02, used_constrained_decoding_rate=1.0),
        ]
        table = sut.results_to_markdown_table(results)
        assert "text" in table and "json_schema" in table
        assert "50%" in table and "100%" in table


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--model", "qwen2.5:1.5b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
