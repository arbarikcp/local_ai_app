import run_benchmark as sut


def test_main_skips_cleanly_when_ollama_unreachable(capsys):
    exit_code = sut.main(["--models", "qwen2.5:1.5b"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.err
