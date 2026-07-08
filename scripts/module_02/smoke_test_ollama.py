"""Lab 2.2 — Run a model through Ollama.

Never installs Ollama or pulls a model. If Ollama isn't reachable, prints
the exact commands to set it up and exits with a clear skip — consistent
with Module 1's honesty rule (no fabricated numbers).

Usage:
    uv run python scripts/module_02/smoke_test_ollama.py [--model qwen2.5:1.5b]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from ollama_probe import (  # noqa: E402
    OllamaUnavailable,
    generate,
    is_ollama_available,
    list_local_models,
)

DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_PROMPT = "Say hello in five words."

INSTALL_INSTRUCTIONS = """\
SKIPPED: Ollama is not reachable at http://localhost:11434.

To complete this lab on a resourced machine:
  brew install ollama
  brew services start ollama          # or: ollama serve
  ollama pull qwen2.5:1.5b
  uv run python scripts/module_02/smoke_test_ollama.py --model qwen2.5:1.5b
"""


def run(model: str, prompt: str) -> int:
    if not is_ollama_available():
        print(INSTALL_INSTRUCTIONS, file=sys.stderr)
        return 1

    models = list_local_models()
    if model not in models:
        print(
            f"SKIPPED: Ollama is running but '{model}' is not pulled. "
            f"Available: {models or '(none)'}. Run: ollama pull {model}",
            file=sys.stderr,
        )
        return 1

    try:
        obs = generate(model, prompt)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print("# Lab 2.2 — Ollama smoke test\n")
    print(f"Model: `{model}`")
    print(f"Prompt: `{prompt}`")
    print(f"Response: {obs.response_text.strip()}")
    print(f"Prompt tokens: {obs.prompt_eval_count}")
    print(f"Output tokens: {obs.eval_count}")
    print(f"TTFT: {obs.ttft_seconds:.2f}s" if obs.ttft_seconds is not None else "TTFT: n/a")
    print(
        f"Tokens/sec: {obs.tokens_per_second:.2f}"
        if obs.tokens_per_second is not None
        else "Tokens/sec: n/a"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)
    return run(args.model, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
