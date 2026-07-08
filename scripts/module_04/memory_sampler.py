"""Process RSS peak sampler, for Lab 4.4 (predict, then measure).

This is real, working measurement tooling — not a stub — proven against a
dummy subprocess in this module's notebook, since there is no model runtime
process on this machine to sample (see the machine constraint in
PROGRESS.md). On a resourced Mac, point it at the Ollama/llama.cpp server
process's pid during a generation call to get the "actual peak memory"
half of Lab 4.4's prediction-vs-actual table.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def find_pid_by_name(process_name: str) -> int | None:
    """Find the first pid whose command matches ``process_name`` via ``pgrep``.

    Used by the Module 4 labs to locate the Ollama/llama.cpp server process
    to sample. Returns None if no matching process is running (e.g. on this
    machine, where no runtime is installed) rather than raising.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", process_name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    first_line = result.stdout.strip().splitlines()[:1]
    if not first_line:
        return None
    try:
        return int(first_line[0])
    except ValueError:
        return None


def get_rss_bytes(pid: int) -> int | None:
    """Current resident set size of ``pid``, in bytes, via macOS/BSD ``ps``.

    Returns None if the process doesn't exist or ``ps`` can't read it
    (already exited, permission denied, etc.) — callers must treat that as
    "no sample," not zero.
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    output = result.stdout.strip()
    if not output:
        return None
    try:
        rss_kib = int(output)
    except ValueError:
        return None
    return rss_kib * 1024


class PeakMemorySampler:
    """Polls a process's RSS in a background thread and tracks the peak."""

    def __init__(self, pid: int, poll_interval_seconds: float = 0.05) -> None:
        self.pid = pid
        self.poll_interval_seconds = poll_interval_seconds
        self._peak_bytes: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = get_rss_bytes(self.pid)
            if sample is not None and (self._peak_bytes is None or sample > self._peak_bytes):
                self._peak_bytes = sample
            time.sleep(self.poll_interval_seconds)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self._peak_bytes

    def __enter__(self) -> PeakMemorySampler:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def peak_bytes(self) -> int | None:
        return self._peak_bytes


def sample_peak_rss_during(
    pid: int, fn: Callable[[], T], poll_interval_seconds: float = 0.05
) -> tuple[T, int | None]:
    """Run ``fn()`` while sampling ``pid``'s RSS peak in the background.

    Returns ``(fn's return value, peak_bytes)``. ``peak_bytes`` is None if
    the process could never be sampled (e.g. it exited before any poll).
    """
    sampler = PeakMemorySampler(pid, poll_interval_seconds)
    sampler.start()
    try:
        result = fn()
    finally:
        peak = sampler.stop()
    return result, peak
