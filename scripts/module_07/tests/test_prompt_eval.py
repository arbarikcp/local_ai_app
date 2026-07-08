import pytest

import prompt_eval as sut
import prompt_variants as pv
from local_ai_core.runtimes.fake import FakeRuntime


class TestLoadRegressionCases:
    def test_loads_the_real_extraction_cases_file(self):
        cases = sut.load_regression_cases()
        assert len(cases) >= 5
        for case in cases:
            assert "id" in case
            assert "text" in case
            assert "schema_keys" in case

    def test_loads_from_a_custom_path(self, tmp_path):
        custom = tmp_path / "cases.jsonl"
        custom.write_text('{"id": "c1", "text": "hi", "schema_keys": ["a"]}\n')
        cases = sut.load_regression_cases(custom)
        assert len(cases) == 1
        assert cases[0]["id"] == "c1"

    def test_skips_blank_lines(self, tmp_path):
        custom = tmp_path / "cases.jsonl"
        custom.write_text('{"id": "c1", "text": "hi", "schema_keys": ["a"]}\n\n\n')
        cases = sut.load_regression_cases(custom)
        assert len(cases) == 1


class TestRunRegressionSuite:
    async def test_marks_cases_passed_when_all_keys_present(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        template = pv.variant_4_with_schema()
        cases = [{"id": "c1", "text": "input text", "schema_keys": ["name", "age", "city"]}]
        results = await sut.run_regression_suite(template, runtime, "m", cases)
        assert results[0].passed is True
        assert results[0].case_id == "c1"

    async def test_marks_cases_failed_when_a_key_is_missing(self):
        runtime = FakeRuntime(default_response='{"name": "X"}')  # missing age, city
        template = pv.variant_4_with_schema()
        cases = [{"id": "c1", "text": "input text", "schema_keys": ["name", "age", "city"]}]
        results = await sut.run_regression_suite(template, runtime, "m", cases)
        assert results[0].passed is False

    async def test_never_asserts_on_exact_output_text(self):
        # Module 6's Gotcha: assert properties (required keys present), not
        # exact strings - two structurally-equivalent-but-differently-worded
        # valid outputs should both pass.
        runtime = FakeRuntime(default_response='{"name": "Alice", "age": 30, "city": "NYC"}')
        template = pv.variant_4_with_schema()
        cases = [{"id": "c1", "text": "text", "schema_keys": ["name", "age", "city"]}]
        results = await sut.run_regression_suite(template, runtime, "m", cases)
        assert results[0].passed is True
        # The check is purely structural (has_required_keys), not a string match.


class TestPassRate:
    def test_computes_fraction_passed(self):
        results = [
            sut.RegressionCaseResult(case_id="a", passed=True, output=""),
            sut.RegressionCaseResult(case_id="b", passed=False, output=""),
            sut.RegressionCaseResult(case_id="c", passed=True, output=""),
        ]
        assert sut.pass_rate(results) == pytest.approx(2 / 3)

    def test_empty_results_is_zero(self):
        assert sut.pass_rate([]) == 0.0


class TestFailingCaseIds:
    def test_lists_only_failed_case_ids(self):
        results = [
            sut.RegressionCaseResult(case_id="a", passed=True, output=""),
            sut.RegressionCaseResult(case_id="b", passed=False, output=""),
        ]
        assert sut.failing_case_ids(results) == ["b"]

    def test_empty_when_all_pass(self):
        results = [sut.RegressionCaseResult(case_id="a", passed=True, output="")]
        assert sut.failing_case_ids(results) == []


class TestCompareCompression:
    async def test_compressed_prompt_is_shorter(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        cases = [{"id": "c1", "text": "input", "schema_keys": ["name", "age", "city"]}]
        result = await sut.compare_compression(
            pv.variant_4_with_schema(), pv.variant_4_compressed(), runtime, "m", cases
        )
        assert result["compressed_prompt_chars"] < result["full_prompt_chars"]

    async def test_both_pass_rates_reported(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        cases = [{"id": "c1", "text": "input", "schema_keys": ["name", "age", "city"]}]
        result = await sut.compare_compression(
            pv.variant_4_with_schema(), pv.variant_4_compressed(), runtime, "m", cases
        )
        assert result["full_pass_rate"] == 1.0
        assert result["compressed_pass_rate"] == 1.0

    async def test_reports_which_cases_newly_fail_under_compression(self):
        runtime = FakeRuntime(default_response='{"name": "X"}')  # always missing fields
        cases = [{"id": "c1", "text": "input", "schema_keys": ["name", "age", "city"]}]
        result = await sut.compare_compression(
            pv.variant_4_with_schema(), pv.variant_4_compressed(), runtime, "m", cases
        )
        assert result["compressed_failing_cases"] == ["c1"]


class TestCompressionResultsToMarkdown:
    def test_renders_all_fields(self):
        results = {
            "full_pass_rate": 1.0,
            "compressed_pass_rate": 0.5,
            "full_prompt_chars": 200,
            "compressed_prompt_chars": 100,
            "compressed_failing_cases": ["c3"],
        }
        md = sut.compression_results_to_markdown(results)
        assert "200 chars" in md
        assert "100 chars" in md
        assert "50%" in md  # 50% reduction
        assert "c3" in md


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--model", "qwen2.5:1.5b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
