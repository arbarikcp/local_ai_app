"""Command safety (ARCHITECTURE.md "Command safety") — `run_tests.py`'s
real invocation is already fixed and not model-controlled (confirmed by
reading the code), so this exists for the one place curriculum's "model
suggests unsafe shell command" failure case could actually reach code: a
hypothetical model-suggested test-run command, checked against an
exact-prefix allowlist before it would ever be allowed to run.
"""

from __future__ import annotations

import sys

ALLOWED_TEST_COMMAND_PREFIX: tuple[str, ...] = (sys.executable, "-m", "pytest")


class UnsafeCommandError(Exception):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(f"command {argv!r} is not an allowed test-run command - refusing to execute")
        self.argv = argv


def validate_test_command(argv: list[str]) -> None:
    """Only a command that starts with exactly `[sys.executable, "-m",
    "pytest"]` is allowed - anything else (a different interpreter, a
    shell pipe, an unrelated binary, extra leading flags) is rejected
    outright. This is a real allowlist check on a real argv list, never a
    shell string - no shell is ever invoked for this comparison or for
    the real test run it gates (`run_tests.py` already uses
    `subprocess.run` with a list argv and no `shell=True`).
    """
    prefix_len = len(ALLOWED_TEST_COMMAND_PREFIX)
    if tuple(argv[:prefix_len]) != ALLOWED_TEST_COMMAND_PREFIX:
        raise UnsafeCommandError(argv)
