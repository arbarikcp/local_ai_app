import lab_4_1_quantization_comparison as sut


def test_rows_to_markdown_table_includes_all_tags_and_handles_missing_peak():
    rows = [
        sut.QuantComparisonRow(
            tag="model-q4",
            response_text="hello",
            prompt_tokens=10,
            output_tokens=5,
            ttft_seconds=0.5,
            tokens_per_second=20.0,
            peak_rss_bytes=2 * 1024**3,
        ),
        sut.QuantComparisonRow(
            tag="model-q8",
            response_text="hi",
            prompt_tokens=10,
            output_tokens=5,
            ttft_seconds=None,
            tokens_per_second=None,
            peak_rss_bytes=None,
        ),
    ]
    table = sut.rows_to_markdown_table(rows)
    assert "model-q4" in table
    assert "model-q8" in table
    assert "2.00 GiB" in table
    assert "n/a" in table


def test_run_lab_skips_memory_sampling_when_no_ollama_process_found(monkeypatch):
    import ollama_probe

    def fake_generate(model, prompt, timeout=300.0):
        return ollama_probe.GenerationObservation(
            model=model,
            prompt=prompt,
            response_text="ok",
            prompt_eval_count=5,
            eval_count=3,
            total_duration_ns=None,
            load_duration_ns=None,
            prompt_eval_duration_ns=None,
            eval_duration_ns=None,
            wall_clock_seconds=0.1,
        )

    monkeypatch.setattr(sut, "generate", fake_generate)
    monkeypatch.setattr(sut, "find_pid_by_name", lambda name: None)

    rows = sut.run_lab(["model-a"], "prompt")
    assert len(rows) == 1
    assert rows[0].peak_rss_bytes is None
    assert rows[0].response_text == "ok"
