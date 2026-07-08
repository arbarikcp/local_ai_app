from lab_1_1_multi_model_run import LabRow, rows_to_markdown_table as rows_to_md_1_1
from lab_1_2_long_prompt_stress_test import StressRow, rows_to_markdown_table as rows_to_md_1_2


def test_lab_1_1_table_includes_all_models_and_handles_missing_metrics():
    rows = [
        LabRow(
            model="qwen2.5:1.5b",
            prompt_tokens=12,
            output_tokens=30,
            ttft_seconds=0.42,
            tokens_per_second=55.5,
            wall_clock_seconds=1.1,
            answer_preview="Some answer",
        ),
        LabRow(
            model="qwen2.5:7b",
            prompt_tokens=12,
            output_tokens=None,
            ttft_seconds=None,
            tokens_per_second=None,
            wall_clock_seconds=4.0,
            answer_preview="",
        ),
    ]
    table = rows_to_md_1_1(rows)
    assert "qwen2.5:1.5b" in table
    assert "qwen2.5:7b" in table
    assert "n/a" in table  # missing metrics rendered explicitly, not blank


def test_lab_1_2_table_flags_failures():
    rows = [
        StressRow(
            target_tokens=500,
            prompt_tokens=510,
            output_tokens=5,
            ttft_seconds=0.3,
            tokens_per_second=40.0,
            wall_clock_seconds=1.0,
            truncated_or_errored=False,
            error=None,
            answer_preview="Paris",
        ),
        StressRow(
            target_tokens=16_000,
            prompt_tokens=None,
            output_tokens=None,
            ttft_seconds=None,
            tokens_per_second=None,
            wall_clock_seconds=0.0,
            truncated_or_errored=True,
            error="context length exceeded",
            answer_preview="",
        ),
    ]
    table = rows_to_md_1_2(rows)
    assert "| 500 |" in table
    assert "| 16000 |" in table
    assert "context length exceeded" in table
    assert table.count("| yes |") == 1
    assert table.count("| no |") == 1
