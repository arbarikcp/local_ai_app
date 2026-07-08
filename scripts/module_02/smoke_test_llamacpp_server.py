"""Lab 2.3 — Run a model through the llama-cpp-python OpenAI-compatible server.

Never installs llama-cpp-python or downloads a GGUF file. This script only
*calls* an already-running server; starting the server is a separate,
explicit step (see docs/modules/02_mac_local_ai_development_environment.md
§8) because launching a model server is not something a smoke test should
do implicitly.

Usage:
    # Terminal 1 (on a resourced Mac, after installing llama-cpp-python[server]):
    uv run python -m llama_cpp.server --model /path/to/model.gguf --port 8080

    # Terminal 2:
    uv run python scripts/module_02/smoke_test_llamacpp_server.py --base-url http://localhost:8080/v1
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_PROMPT = "Say hello in five words."

INSTALL_INSTRUCTIONS = """\
SKIPPED: could not reach a llama-cpp-python server at {base_url}.

To complete this lab on a resourced machine:
  CMAKE_ARGS="-DGGML_METAL=ON" uv add "llama-cpp-python[server]"
  uv add openai   # OpenAI-compatible client, not a hosted-API dependency
  uv run python -m llama_cpp.server --model /path/to/model.gguf --port 8080
  uv run python scripts/module_02/smoke_test_llamacpp_server.py
"""


def check_openai_client_importable() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def run(base_url: str, prompt: str, model: str) -> int:
    if not check_openai_client_importable():
        print(
            "SKIPPED: the `openai` package is not installed. Run: uv add openai",
            file=sys.stderr,
        )
        return 1

    from openai import APIConnectionError, OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed-for-local-server")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except APIConnectionError:
        print(INSTALL_INSTRUCTIONS.format(base_url=base_url), file=sys.stderr)
        return 1

    print("# Lab 2.3 — llama-cpp-python server smoke test\n")
    print(f"Base URL: `{base_url}`")
    print(f"Prompt: `{prompt}`")
    print(f"Response: {response.choices[0].message.content}")
    if response.usage is not None:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="local-gguf-model")
    args = parser.parse_args(argv)
    return run(args.base_url, args.prompt, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
