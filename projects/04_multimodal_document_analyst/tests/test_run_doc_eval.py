import pytest

from run_doc_eval import GOLDEN_SET_PATH, load_golden_set, run_eval


class TestLoadGoldenSet:
    def test_loads_three_pages_and_three_questions(self):
        pages, questions = load_golden_set(GOLDEN_SET_PATH)
        assert len(pages) == 3
        assert len(questions) == 3
        assert pages[2].expected_route == "vlm"
        assert questions[0].expected_page_id == "multi_page_form::page2"


@pytest.mark.asyncio
class TestRunEval:
    async def test_a_correctly_scripted_run_scores_perfectly(self):
        pages, questions = load_golden_set(GOLDEN_SET_PATH)
        summary, page_results, question_results = await run_eval(pages, questions)

        assert summary.total_pages == 3
        assert summary.route_accuracy == 1.0
        assert summary.text_layer_fidelity == 1.0
        assert summary.mean_field_exact_match == 1.0
        assert summary.total_questions == 3
        assert summary.citation_correctness_rate == 1.0
        assert summary.citation_verification_rate == 1.0
        assert summary.answer_correctness_rate == 1.0
        assert summary.peak_rss_mb > 0

    async def test_per_page_results_report_the_vlm_page_with_no_field_score(self):
        pages, questions = load_golden_set(GOLDEN_SET_PATH)
        _, page_results, _ = await run_eval(pages, questions)

        page3 = next(r for r in page_results if r.page_id == "multi_page_form::page3")
        assert page3.route_correct is True
        assert page3.field_exact_match is None
