import adapter_packaging_demo as sut


class TestRunLab:
    def test_lora_params_are_a_genuine_reduction(self):
        result = sut.run_lab()
        assert result["lora_trainable_params"] < result["full_finetune_params"]
        assert 0 < result["reduction_ratio"] < 1

    def test_dataset_hash_is_deterministic(self):
        result_a = sut.run_lab()
        result_b = sut.run_lab()
        assert result_a["dataset_hash"] == result_b["dataset_hash"]

    def test_adapter_is_registered(self):
        result = sut.run_lab()
        assert result["registered_adapter_name"] == "ticket-classifier-v1"
        assert result["registered_adapter_created_at"] is not None

    def test_synthetic_loss_curve_is_flagged_as_overfitting(self):
        result = sut.run_lab()
        assert result["overfitting_detected"] is True
        assert result["stop_at_epoch"] is not None
        assert result["best_validation_epoch"] == 3


class TestResultToMarkdown:
    def test_markdown_includes_failure_analysis(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Failure mode" in markdown
        assert "Overfitting detected: True" in markdown
