import pytest

import lab_4_4_predict_then_measure as sut


def test_predict_and_measure_computes_predictions_and_calls_generate(monkeypatch):
    import ollama_probe

    calls = []

    def fake_generate(model, prompt, timeout=600.0):
        calls.append((model, prompt))
        return ollama_probe.GenerationObservation(
            model=model, prompt=prompt, response_text="ok", prompt_eval_count=1,
            eval_count=1, total_duration_ns=None, load_duration_ns=None,
            prompt_eval_duration_ns=None, eval_duration_ns=None, wall_clock_seconds=0.1,
        )

    monkeypatch.setattr(sut, "generate", fake_generate)
    monkeypatch.setattr(sut, "find_pid_by_name", lambda name: None)

    rows = sut.predict_and_measure(
        model_tag="qwen2.5:7b-instruct-q4_K_M",
        shape_id="qwen2.5-7b",
        quant="Q4_K_M",
        context_lengths=[2000, 8000],
    )
    assert len(rows) == 2
    assert len(calls) == 2
    assert rows[0].context_tokens == 2000
    assert rows[0].predicted_low_gib > 0
    assert rows[0].predicted_high_gib > rows[0].predicted_low_gib
    assert rows[0].actual_peak_gib is None  # no ollama pid found -> not measured
    assert "fill in manually" in rows[0].gap_explanation


def test_predict_and_measure_raises_for_unknown_shape(monkeypatch):
    with pytest.raises(KeyError):
        sut.predict_and_measure(
            model_tag="x", shape_id="not-a-real-shape", quant="Q4_K_M", context_lengths=[2000]
        )


def test_rows_to_markdown_table_shows_not_measured_when_no_actual_value():
    row = sut.PredictVsActualRow(
        model_tag="m", quant="Q4_K_M", context_tokens=2000, predicted_low_gib=4.0,
        predicted_high_gib=5.5, actual_peak_gib=None, gap_explanation="explain here",
    )
    table = sut.rows_to_markdown_table([row])
    assert "not measured" in table
    assert "4.0-5.5 GB/GiB" in table
    assert "explain here" in table


def test_rows_to_markdown_table_shows_measured_actual_value():
    row = sut.PredictVsActualRow(
        model_tag="m", quant="Q4_K_M", context_tokens=2000, predicted_low_gib=4.0,
        predicted_high_gib=5.5, actual_peak_gib=6.25, gap_explanation="explain here",
    )
    table = sut.rows_to_markdown_table([row])
    assert "6.25 GiB" in table
