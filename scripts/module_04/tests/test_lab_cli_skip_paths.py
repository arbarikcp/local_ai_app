"""Every Module 4 lab must skip cleanly (not crash) when Ollama is
unreachable, which is the actual state of this machine — see the machine
constraint in PROGRESS.md. These tests exercise that real code path.
"""

import lab_4_1_quantization_comparison as lab41
import lab_4_2_context_scaling as lab42
import lab_4_3_concurrency_simulation as lab43
import lab_4_4_predict_then_measure as lab44


def test_lab_4_1_main_skips_cleanly(capsys):
    exit_code = lab41.main(["--tags", "qwen2.5:7b-instruct-q4_K_M"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err


def test_lab_4_2_main_skips_cleanly(capsys):
    exit_code = lab42.main(["--model", "qwen2.5:3b"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err


def test_lab_4_3_main_skips_cleanly(capsys):
    exit_code = lab43.main(["--model", "qwen2.5:3b"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err


def test_lab_4_4_main_skips_cleanly(capsys):
    exit_code = lab44.main(["--model-tag", "qwen2.5:7b-instruct-q4_K_M", "--shape", "qwen2.5-7b"])
    assert exit_code == 1
    assert "SKIPPED" in capsys.readouterr().err
