import evaluation_demo as sut


class TestRunLab:
    async def test_fine_tuned_candidate_scores_perfectly_against_prompt_only(self):
        result = await sut.run_lab()
        before_after = result["prompt_only_vs_fine_tuned"]
        assert before_after.candidate_mean_score == 1.0
        assert before_after.candidate_improved is True

    async def test_fine_tuned_candidate_also_improves_on_an_unhelpful_baseline(self):
        result = await sut.run_lab()
        vs_unhelpful = result["unhelpful_vs_fine_tuned"]
        assert vs_unhelpful.baseline_mean_score == 0.0
        assert vs_unhelpful.candidate_mean_score == 1.0

    async def test_prompt_only_baseline_is_genuinely_worse_than_candidate(self):
        result = await sut.run_lab()
        before_after = result["prompt_only_vs_fine_tuned"]
        assert before_after.baseline_mean_score < before_after.candidate_mean_score


class TestResultToMarkdown:
    async def test_markdown_reports_both_comparisons(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Lab 3" in markdown
        assert "Lab 4" in markdown
        assert "improved=True" in markdown
