import platform

import smoke_test_mlx as sut


def test_is_apple_silicon_matches_platform_machine():
    assert sut.is_apple_silicon() == (platform.machine() == "arm64")


def test_check_mlx_importable_is_a_bool():
    assert isinstance(sut.check_mlx_importable(), bool)


def test_run_skips_with_clear_reason_on_this_machine(capsys):
    exit_code = sut.run(model="mlx-community/does-not-matter", prompt="hi")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert exit_code == 1
    assert "SKIPPED" in combined
    if sut.is_apple_silicon():
        # mlx_lm is not an installed dependency of this course repo (see
        # module constraint: no model runtimes installed on this machine).
        assert "mlx_lm" in combined
    else:
        assert "Apple Silicon" in combined
