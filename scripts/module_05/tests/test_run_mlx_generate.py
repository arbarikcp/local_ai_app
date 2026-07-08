import pytest

import run_mlx_generate as sut


def _make_summary(**overrides):
    defaults = dict(
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        load_seconds=2.0,
        cold_seconds=1.0,
        warm_seconds=0.2,
        stream_ttft_seconds=0.15,
        stream_total_seconds=0.9,
        cold_text="cold response",
        warm_text="warm response",
        streamed_text="streamed response",
    )
    defaults.update(overrides)
    return sut.build_summary(**defaults)


class TestWarmupSpeedupFactor:
    def test_computed_as_cold_over_warm(self):
        summary = _make_summary(cold_seconds=1.0, warm_seconds=0.25)
        assert summary.warmup_speedup_factor == pytest.approx(4.0)

    def test_none_when_warm_seconds_is_zero(self):
        summary = _make_summary(cold_seconds=1.0, warm_seconds=0.0)
        assert summary.warmup_speedup_factor is None


class TestSummaryToMarkdown:
    def test_renders_every_field_when_all_present(self):
        summary = _make_summary()
        md = sut.summary_to_markdown(summary)
        assert "mlx-community/Qwen2.5-1.5B-Instruct-4bit" in md
        assert "Load time: 2.00s" in md
        assert "Cold generate: 1.00s" in md
        assert "Warm generate: 0.20s" in md
        assert "5.00x" in md
        assert "0.150s" in md
        assert "0.90s" in md

    def test_does_not_collapse_when_stream_total_is_none(self):
        # Regression test: an earlier version of summary_to_markdown had a
        # string-concatenation/ternary bug that silently dropped every line
        # except "Streaming total" whenever stream_total_seconds was None.
        summary = _make_summary(stream_ttft_seconds=None, stream_total_seconds=None)
        md = sut.summary_to_markdown(summary)
        assert "Load time: 2.00s" in md
        assert "Cold generate: 1.00s" in md
        assert "Warm generate: 0.20s" in md
        assert "Warmup speedup" in md
        assert "Streaming TTFT: n/a" in md
        assert "Streaming total: n/a" in md

    def test_handles_none_speedup_gracefully(self):
        summary = _make_summary(warm_seconds=0.0)
        md = sut.summary_to_markdown(summary)
        assert "n/a" in md


def test_build_summary_round_trips_all_fields():
    summary = _make_summary(model_id="some-model")
    assert summary.model_id == "some-model"
    assert summary.cold_text == "cold response"
    assert summary.warm_text == "warm response"
    assert summary.streamed_text == "streamed response"


def test_main_skips_on_non_apple_silicon(monkeypatch, capsys):
    monkeypatch.setattr(sut, "is_apple_silicon", lambda: False)
    exit_code = sut.main(["--model", "some-model"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err


def test_main_skips_when_mlx_not_importable(monkeypatch, capsys):
    monkeypatch.setattr(sut, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(sut, "check_mlx_importable", lambda: False)
    exit_code = sut.main(["--model", "some-model"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err
