import pytest

import smoke_test_ollama


def test_run_skips_cleanly_when_ollama_unreachable(capsys):
    # No Ollama server is expected to be running in the test environment.
    exit_code = smoke_test_ollama.run(model="qwen2.5:1.5b", prompt="hi")
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.err
    assert "brew install ollama" in captured.err


def test_run_returns_nonzero_and_does_not_raise():
    # The whole point of a smoke test is that "runtime missing" is a clean,
    # typed failure path, not an unhandled exception.
    try:
        code = smoke_test_ollama.run(model="qwen2.5:1.5b", prompt="hi")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"smoke test raised instead of returning a skip code: {exc}")
    assert isinstance(code, int)
    assert code != 0
