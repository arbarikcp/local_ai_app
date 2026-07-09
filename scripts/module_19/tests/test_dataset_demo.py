import dataset_demo as sut


class TestRunLab:
    def test_deliberate_duplicate_is_the_only_dropped_example(self):
        result = sut.run_lab()
        assert result["raw_example_count"] == 41
        assert result["dropped_count"] == 1
        assert result["dropped_reasons"] == ["duplicate instruction+input"]

    def test_cleaned_dataset_matches_the_committed_forty_examples(self):
        result = sut.run_lab()
        assert result["cleaned_example_count"] == 40

    def test_split_sizes_match_the_configured_ratios(self):
        result = sut.run_lab()
        assert result["train_count"] == 32
        assert result["validation_count"] == 4
        assert result["test_count"] == 4

    def test_split_has_no_leakage(self):
        result = sut.run_lab()
        assert result["leak_count"] == 0


class TestResultToMarkdown:
    def test_markdown_mentions_the_split_sizes(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "train=32" in markdown
        assert "0 leaked" in markdown
