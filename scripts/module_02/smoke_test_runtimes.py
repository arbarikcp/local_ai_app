"""Module 2 deliverable generator — runs every check available on this
machine and renders one combined markdown report.

Never installs anything. Sections for runtimes that aren't available print
an honest "not available on this machine" note with the exact setup
commands, rather than being silently omitted.

Usage:
    uv run python scripts/module_02/smoke_test_runtimes.py > /tmp/module_02_raw.md
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mac_environment_check  # noqa: E402
import model_cache  # noqa: E402
import smoke_test_llamacpp_server  # noqa: E402
import smoke_test_mlx  # noqa: E402
import smoke_test_ollama  # noqa: E402


def _capture(fn, *args, **kwargs) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            code = fn(*args, **kwargs)
        except SystemExit as exc:  # some entry points may call sys.exit indirectly
            code = exc.code or 0
    return code, buf.getvalue()


def build_report() -> str:
    sections = ["# Module 2 — environment smoke test report\n"]

    profile = mac_environment_check.detect_machine_profile()
    checks = mac_environment_check.check_all_tools()
    sections.append("## Machine profile\n")
    sections.append(
        f"- Architecture: {profile.architecture} "
        f"({'Apple Silicon' if profile.is_apple_silicon else 'not Apple Silicon'})\n"
        f"- macOS version: {profile.macos_version}\n"
        f"- Python version: {profile.python_version}\n"
    )
    sections.append("## Dev tool check\n")
    sections.append(mac_environment_check.tools_to_markdown_table(checks) + "\n")

    sections.append("## Model cache report\n")
    cache_locations = model_cache.scan_caches()
    sections.append(model_cache.caches_to_markdown_table(cache_locations) + "\n")

    sections.append("## Lab 2.2 — Ollama\n")
    code, output = _capture(smoke_test_ollama.run, smoke_test_ollama.DEFAULT_MODEL, smoke_test_ollama.DEFAULT_PROMPT)
    sections.append(f"Exit code: {code}\n\n```text\n{output.strip()}\n```\n")

    sections.append("## Lab 2.3 — llama-cpp-python server\n")
    code, output = _capture(
        smoke_test_llamacpp_server.run,
        smoke_test_llamacpp_server.DEFAULT_BASE_URL,
        smoke_test_llamacpp_server.DEFAULT_PROMPT,
        "local-gguf-model",
    )
    sections.append(f"Exit code: {code}\n\n```text\n{output.strip()}\n```\n")

    sections.append("## Lab 2.4 — MLX\n")
    code, output = _capture(smoke_test_mlx.run, smoke_test_mlx.DEFAULT_MODEL, smoke_test_mlx.DEFAULT_PROMPT)
    sections.append(f"Exit code: {code}\n\n```text\n{output.strip()}\n```\n")

    return "\n".join(sections)


def main() -> int:
    print(build_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
