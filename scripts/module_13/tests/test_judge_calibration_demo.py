import judge_calibration_demo as sut


class TestRunJudgeCalibration:
    async def test_produces_one_label_pair_per_case(self):
        result = await sut.run_judge_calibration()
        assert len(result["judge_labels"]) == len(sut.CALIBRATION_CASES)
        assert len(result["human_labels"]) == len(sut.CALIBRATION_CASES)

    async def test_agreement_metrics_are_in_valid_ranges(self):
        result = await sut.run_judge_calibration()
        assert 0.0 <= result["simple_agreement"] <= 1.0
        assert -1.0 <= result["cohens_kappa"] <= 1.0

    async def test_agreement_is_not_trivially_perfect(self):
        # The calibration set includes deliberately ambiguous cases so this
        # lab has something real to report, not a rubber-stamped 1.0.
        result = await sut.run_judge_calibration()
        assert result["simple_agreement"] < 1.0


class TestRunHallucinationAuroc:
    def test_produces_one_label_and_score_per_case(self):
        result = sut.run_hallucination_auroc()
        assert len(result["labels"]) == len(sut.CALIBRATION_CASES)
        assert len(result["scores"]) == len(sut.CALIBRATION_CASES)

    def test_auroc_is_a_valid_probability(self):
        result = sut.run_hallucination_auroc()
        assert 0.0 <= result["auroc"] <= 1.0

    def test_auroc_beats_chance_on_this_calibration_set(self):
        result = sut.run_hallucination_auroc()
        assert result["auroc"] > 0.5


class TestResultToMarkdown:
    async def test_includes_both_labs(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "kappa" in markdown
        assert "AUROC" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Cohen's kappa" in captured.out
