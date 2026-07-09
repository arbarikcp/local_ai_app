from local_ai_core.finetuning.dataset import (
    TrainingExample,
    clean_dataset,
    detect_leakage,
    load_jsonl,
    split_dataset,
)


def make_example(i: int, output: str = "answer") -> TrainingExample:
    return TrainingExample(instruction="classify", input=f"ticket {i}", output=output)


class TestLoadJsonl:
    def test_loads_the_real_committed_dataset(self):
        path = "datasets/finetuning/ticket_classification.jsonl"
        examples = load_jsonl(path)
        assert len(examples) == 40
        assert all(isinstance(e, TrainingExample) for e in examples)
        assert {e.output for e in examples} == {"account", "billing", "technical", "security"}

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"instruction": "a", "input": "b", "output": "c"}\n\n'
            '{"instruction": "d", "input": "e", "output": "f"}\n'
        )
        examples = load_jsonl(path)
        assert len(examples) == 2


class TestCleanDataset:
    def test_drops_exact_duplicate_instruction_and_input(self):
        examples = [make_example(1), make_example(1)]
        result = clean_dataset(examples)
        assert len(result.kept) == 1
        assert len(result.dropped) == 1
        assert result.dropped[0].reason == "duplicate instruction+input"

    def test_drops_examples_with_output_shorter_than_min(self):
        examples = [make_example(1, output="")]
        result = clean_dataset(examples, min_output_chars=1)
        assert result.kept == []
        assert result.dropped[0].reason == "output shorter than min_output_chars"

    def test_drops_examples_with_input_longer_than_max(self):
        example = TrainingExample(instruction="classify", input="x" * 50, output="answer")
        result = clean_dataset([example], max_input_chars=10)
        assert result.kept == []
        assert result.dropped[0].reason == "input longer than max_input_chars"

    def test_keeps_valid_unique_examples(self):
        examples = [make_example(1), make_example(2)]
        result = clean_dataset(examples)
        assert result.kept == examples
        assert result.dropped == []


class TestSplitDataset:
    def test_split_sizes_cover_the_full_dataset(self):
        examples = [make_example(i) for i in range(100)]
        split = split_dataset(examples, train_ratio=0.8, validation_ratio=0.1, seed=42)
        assert len(split.train) + len(split.validation) + len(split.test) == 100
        assert len(split.train) == 80
        assert len(split.validation) == 10
        assert len(split.test) == 10

    def test_same_seed_produces_the_same_split(self):
        examples = [make_example(i) for i in range(20)]
        split_a = split_dataset(examples, seed=7)
        split_b = split_dataset(examples, seed=7)
        assert split_a.train == split_b.train
        assert split_a.validation == split_b.validation
        assert split_a.test == split_b.test

    def test_invalid_ratios_raise(self):
        examples = [make_example(i) for i in range(10)]
        try:
            split_dataset(examples, train_ratio=0.9, validation_ratio=0.2)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestDetectLeakage:
    def test_clean_split_reports_no_leakage(self):
        examples = [make_example(i) for i in range(30)]
        split = split_dataset(examples, seed=1)
        assert detect_leakage(split) == []

    def test_deliberately_leaked_split_is_caught(self):
        from local_ai_core.finetuning.dataset import DatasetSplit

        leaked_example = make_example(1)
        split = DatasetSplit(
            train=[leaked_example],
            validation=[leaked_example],
            test=[make_example(2)],
        )
        leaks = detect_leakage(split)
        assert leaked_example.dedup_key() in leaks
