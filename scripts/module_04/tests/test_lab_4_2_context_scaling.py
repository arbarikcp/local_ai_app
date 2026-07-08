import lab_4_2_context_scaling as sut


def test_rows_to_markdown_table_shows_errors_and_missing_peak():
    rows = [
        sut.ContextScalingRow(
            target_tokens=2000, actual_prompt_tokens=2010, wall_clock_seconds=1.5,
            peak_rss_bytes=3 * 1024**3, error=None,
        ),
        sut.ContextScalingRow(
            target_tokens=16000, actual_prompt_tokens=None, wall_clock_seconds=0.0,
            peak_rss_bytes=None, error="context length exceeded",
        ),
    ]
    table = sut.rows_to_markdown_table(rows)
    assert "| 2000 |" in table
    assert "| 16000 |" in table
    assert "3.00 GiB" in table
    assert "context length exceeded" in table


def test_run_lab_records_exceptions_per_context_length_without_aborting_the_sweep(monkeypatch):
    import ollama_probe

    calls = []

    def fake_generate(model, prompt, timeout=300.0):
        calls.append(prompt)
        if len(calls) == 2:
            raise ollama_probe.OllamaUnavailable("simulated failure")
        return ollama_probe.GenerationObservation(
            model=model, prompt=prompt, response_text="ok", prompt_eval_count=1,
            eval_count=1, total_duration_ns=None, load_duration_ns=None,
            prompt_eval_duration_ns=None, eval_duration_ns=None, wall_clock_seconds=0.1,
        )

    monkeypatch.setattr(sut, "generate", fake_generate)
    monkeypatch.setattr(sut, "find_pid_by_name", lambda name: None)

    rows = sut.run_lab("fake-model", [500, 1000, 2000])
    assert len(rows) == 3
    assert rows[0].error is None
    assert rows[1].error == "simulated failure"
    assert rows[2].error is None
