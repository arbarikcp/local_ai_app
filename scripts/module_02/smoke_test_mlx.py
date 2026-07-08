"""Lab 2.4 — Run a model through MLX.

Never installs mlx/mlx-lm and never downloads a model. Apple Silicon only
(see docs/modules/02_mac_local_ai_development_environment.md §1) — checks
architecture before attempting anything.

Usage:
    uv run python scripts/module_02/smoke_test_mlx.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_PROMPT = "Say hello in five words."

NOT_APPLE_SILICON_MESSAGE = """\
SKIPPED: MLX requires Apple Silicon (arm64); this machine reports '{arch}'.
MLX is not available on Intel Macs — use Ollama or llama.cpp instead for this machine.
"""

INSTALL_INSTRUCTIONS = """\
SKIPPED: `mlx_lm` is not importable.

To complete this lab on a resourced Apple Silicon Mac:
  uv add mlx mlx-lm
  uv run python scripts/module_02/smoke_test_mlx.py

Note: the first run downloads the model from Hugging Face into
~/.cache/huggingface/hub — see model_cache.py for sizing that cache.
"""


def is_apple_silicon() -> bool:
    return platform.machine() == "arm64"


def check_mlx_importable() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def run(model: str, prompt: str) -> int:
    if not is_apple_silicon():
        print(NOT_APPLE_SILICON_MESSAGE.format(arch=platform.machine()), file=sys.stderr)
        return 1

    if not check_mlx_importable():
        print(INSTALL_INSTRUCTIONS, file=sys.stderr)
        return 1

    from mlx_lm import generate, load

    start = time.perf_counter()
    mlx_model, tokenizer = load(model)
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    text = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=64)
    generate_seconds = time.perf_counter() - start

    print("# Lab 2.4 — MLX smoke test\n")
    print(f"Model: `{model}`")
    print(f"Prompt: `{prompt}`")
    print(f"Response: {text}")
    print(f"Load time: {load_seconds:.2f}s")
    print(f"Generation wall clock: {generate_seconds:.2f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)
    return run(args.model, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
