#!/usr/bin/env bash
# Module 5 Lab 1 — start an OpenAI-compatible llama.cpp-family server
# through a repeatable command.
#
# Documentation-as-script, same pattern as scripts/module_02/setup_mac.sh:
# reviewed and understood before running, NOT executed as part of this
# repo's build/test process, and must never be run on a machine that
# shouldn't have a model runtime installed.
#
# Supports two backends - pick one via the first argument:
#   bash scripts/module_05/serve_llamacpp.sh native   /path/to/model.gguf
#   bash scripts/module_05/serve_llamacpp.sh python    /path/to/model.gguf
#
# "native" uses the llama.cpp repo's own `llama-server` binary (built with
# Metal per Module 2 §7). "python" uses `llama-cpp-python[server]`'s
# `python -m llama_cpp.server` entrypoint (Module 2 §8). Both expose an
# OpenAI-compatible /v1/chat/completions endpoint.

set -euo pipefail

BACKEND="${1:?Usage: $0 <native|python> <path-to-gguf-model>}"
MODEL_PATH="${2:?Usage: $0 <native|python> <path-to-gguf-model>}"
PORT="${PORT:-8080}"
BASE_URL="http://localhost:${PORT}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
  echo "A server is already responding at ${BASE_URL}. Stop it first, or set PORT to a free port."
  exit 0
fi

case "${BACKEND}" in
  native)
    echo "Starting llama.cpp's native llama-server on port ${PORT}..."
    echo "(requires llama.cpp built with Metal - see Module 2 theory doc §7)"
    ./llama.cpp/build/bin/llama-server --model "${MODEL_PATH}" --port "${PORT}" &
    ;;
  python)
    echo "Starting llama-cpp-python's OpenAI-compatible server on port ${PORT}..."
    echo "(requires: CMAKE_ARGS=\"-DGGML_METAL=ON\" uv add \"llama-cpp-python[server]\" - Module 2 theory doc §8)"
    uv run python -m llama_cpp.server --model "${MODEL_PATH}" --port "${PORT}" &
    ;;
  *)
    echo "Unknown backend '${BACKEND}'. Use 'native' or 'python'." >&2
    exit 1
    ;;
esac

SERVER_PID=$!
echo "Server starting (pid ${SERVER_PID}). Waiting for it to become reachable..."
for _ in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
  echo "Server did not become reachable within 60s (model load can be slow for larger models)." >&2
  echo "Check that the process (pid ${SERVER_PID}) is still running." >&2
  exit 1
fi

echo "Ready at ${BASE_URL}. Verify with:"
echo "  uv run python scripts/module_05/llamacpp_openai_streaming.py --base-url ${BASE_URL}/v1"
