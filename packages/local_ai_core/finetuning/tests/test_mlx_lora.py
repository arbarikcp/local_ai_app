from local_ai_core.finetuning.mlx_lora import (
    MlxLoraTrainer,
    TrainingConfig,
    is_mlx_lm_available,
)


class FakeLoraTrainer:
    def __init__(self, *, train_exit_code: int = 0, merge_exit_code: int = 0) -> None:
        self.train_exit_code = train_exit_code
        self.merge_exit_code = merge_exit_code
        self.train_calls: list[TrainingConfig] = []
        self.merge_calls: list[tuple[str, str, str]] = []

    def train(self, config: TrainingConfig) -> int:
        self.train_calls.append(config)
        return self.train_exit_code

    def merge(self, base_model: str, adapter_dir: str, merged_output_dir: str) -> int:
        self.merge_calls.append((base_model, adapter_dir, merged_output_dir))
        return self.merge_exit_code


class TestTrainLoraAdapter:
    def test_successful_training_run_reports_success(self):
        fake = FakeLoraTrainer(train_exit_code=0)
        trainer = MlxLoraTrainer(train_fn=fake.train)
        config = TrainingConfig(
            base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            dataset_dir="datasets/finetuning",
            adapter_output_dir="adapters/ticket-classifier-v1",
        )
        result = trainer.train_lora_adapter(config)
        assert result.succeeded is True
        assert result.adapter_output_dir == "adapters/ticket-classifier-v1"
        assert fake.train_calls == [config]

    def test_failed_training_run_reports_failure(self):
        fake = FakeLoraTrainer(train_exit_code=1)
        trainer = MlxLoraTrainer(train_fn=fake.train)
        config = TrainingConfig(
            base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            dataset_dir="datasets/finetuning",
            adapter_output_dir="adapters/ticket-classifier-v1",
        )
        result = trainer.train_lora_adapter(config)
        assert result.succeeded is False


class TestMergeAdapter:
    def test_successful_merge_reports_success(self):
        fake = FakeLoraTrainer(merge_exit_code=0)
        trainer = MlxLoraTrainer(merge_fn=fake.merge)
        result = trainer.merge_adapter(
            base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            adapter_dir="adapters/ticket-classifier-v1",
            merged_output_dir="merged/ticket-classifier-v1",
        )
        assert result.succeeded is True
        assert result.merged_model_dir == "merged/ticket-classifier-v1"
        assert fake.merge_calls == [
            (
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "adapters/ticket-classifier-v1",
                "merged/ticket-classifier-v1",
            )
        ]

    def test_failed_merge_reports_failure(self):
        fake = FakeLoraTrainer(merge_exit_code=1)
        trainer = MlxLoraTrainer(merge_fn=fake.merge)
        result = trainer.merge_adapter(
            base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            adapter_dir="adapters/ticket-classifier-v1",
            merged_output_dir="merged/ticket-classifier-v1",
        )
        assert result.succeeded is False


class TestHonestSkipForRealExecution:
    def test_mlx_lm_is_not_available_on_this_machine(self):
        # This repo's dev machine intentionally never has an LLM runtime
        # installed - real training/merging is honest-skipped here, proven
        # by mlx-lm genuinely not being importable rather than mocked away.
        assert is_mlx_lm_available() is False
