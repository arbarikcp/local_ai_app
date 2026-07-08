import pytest

import extraction_eval as sut
from local_ai_core.extraction.pipeline import ExtractionPipeline
from local_ai_core.extraction.schemas import PersonExtraction
from local_ai_core.runtimes.fake import FakeRuntime


class TestEvaluateAgainstGoldenLabels:
    async def test_perfect_pipeline_gives_full_accuracy_and_no_review(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        cases = [{"id": "c1", "text": "text", "reference": {"name": "Maria", "age": 29, "city": "Austin"}}]
        summary = await sut.evaluate_against_golden_labels(pipeline, "m", cases)
        assert summary.mean_field_accuracy == 1.0
        assert summary.review_rate == 0.0
        assert len(summary.review_queue) == 0

    async def test_imperfect_pipeline_produces_partial_accuracy_and_review_items(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": null, "city": null}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        cases = [{"id": "c1", "text": "text", "reference": {"name": "Maria", "age": 29, "city": "Austin"}}]
        summary = await sut.evaluate_against_golden_labels(pipeline, "m", cases)
        assert summary.mean_field_accuracy == pytest.approx(1 / 3)
        assert summary.case_results[0].confidence == "medium"  # one missing-field risk factor

    async def test_produces_one_case_result_per_case(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        cases = [
            {"id": "c1", "text": "t1", "reference": {"name": "X", "age": 1, "city": "Y"}},
            {"id": "c2", "text": "t2", "reference": {"name": "X", "age": 1, "city": "Y"}},
        ]
        summary = await sut.evaluate_against_golden_labels(pipeline, "m", cases)
        assert [r.case_id for r in summary.case_results] == ["c1", "c2"]

    async def test_shares_the_pipelines_review_queue_not_a_new_one(self):
        runtime = FakeRuntime(default_response="not json")
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"], max_repair_attempts=0)
        cases = [{"id": "c1", "text": "text", "reference": {"name": "X", "age": 1, "city": "Y"}}]
        summary = await sut.evaluate_against_golden_labels(pipeline, "m", cases)
        assert summary.review_queue is pipeline.review_queue
        assert len(summary.review_queue) == 1


class TestGoldenEvalSummaryProperties:
    def test_mean_field_accuracy_empty_is_zero(self):
        from local_ai_core.extraction.review_queue import ReviewQueue

        summary = sut.GoldenEvalSummary(case_results=[], review_queue=ReviewQueue())
        assert summary.mean_field_accuracy == 0.0
        assert summary.review_rate == 0.0


class TestSummaryToMarkdown:
    def test_renders_all_cases_and_footer_stats(self):
        from local_ai_core.extraction.review_queue import ReviewQueue

        results = [
            sut.GoldenEvalCaseResult(case_id="c1", field_accuracy=1.0, confidence="high", needs_review=False),
            sut.GoldenEvalCaseResult(case_id="c2", field_accuracy=0.5, confidence="medium", needs_review=True),
        ]
        summary = sut.GoldenEvalSummary(case_results=results, review_queue=ReviewQueue())
        md = sut.summary_to_markdown(summary)
        assert "c1" in md and "c2" in md
        assert "Mean field accuracy: 75%" in md
        assert "Review rate: 50%" in md


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--model", "qwen2.5:1.5b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
