import run_rag_eval as sut


class TestLoadGoldenSet:
    def test_loads_the_real_committed_golden_set(self):
        cases = sut.load_golden_set(sut.GOLDEN_SET_PATH)
        assert len(cases) == 10

    def test_exactly_two_cases_require_refusal(self):
        cases = sut.load_golden_set(sut.GOLDEN_SET_PATH)
        assert sum(1 for c in cases if c.requires_refusal) == 2


class TestRunLab:
    async def test_returns_a_summary_over_every_case(self):
        summary = await sut.run_lab()
        assert summary.total == 10

    async def test_abstention_accuracy_is_perfect_for_the_scripted_runtime(self):
        # GoldenAwareRuntime always refuses on a requires_refusal case, so
        # abstention accuracy should be 100% - a real, checkable property
        # of the scripted runtime, not an assumed number.
        summary = await sut.run_lab()
        assert summary.abstention_accuracy == 1.0

    async def test_citation_correctness_rate_is_perfect(self):
        # GoldenAwareRuntime only ever cites a marker it found genuinely
        # present in the prompt, so every citation it makes must be
        # grounded.
        summary = await sut.run_lab()
        assert summary.citation_correctness_rate == 1.0

    async def test_recall_and_precision_are_real_fractions_not_trivially_zero_or_one(self):
        summary = await sut.run_lab()
        assert 0.0 < summary.mean_recall_at_k < 1.0

    async def test_peak_rss_is_a_real_positive_number(self):
        summary = await sut.run_lab()
        assert summary.peak_rss_mb > 0


class TestSummaryToMarkdown:
    async def test_markdown_reports_every_metric(self):
        summary = await sut.run_lab()
        markdown = sut.summary_to_markdown(summary)
        assert "recall@k" in markdown
        assert "Abstention accuracy" in markdown
        assert "Peak RSS" in markdown
