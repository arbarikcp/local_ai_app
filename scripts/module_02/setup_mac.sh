#!/usr/bin/env bash
# Module 2 — recommended Mac setup for local AI development.
#
# This script is documentation you run deliberately on a machine you intend
# to use for local model work. It is NOT executed as part of this repo's
# build/test process, and must never be run on a machine that shouldn't have
# model runtimes or model weights installed (see
# docs/modules/02_mac_local_ai_development_environment.md).
#
# Usage (on a resourced Mac, after reading it top to bottom):
#   bash scripts/module_02/setup_mac.sh

set -euo pipefail

echo "== Xcode Command Line Tools =="
xcode-select --install || echo "(already installed, or install triggered — follow the GUI prompt)"

echo "== Homebrew base tools =="
brew install git make cmake python@3.12 uv jq ripgrep

echo "== Ollama =="
brew install ollama
echo "Start it with: brew services start ollama   (or: ollama serve)"
echo "Then pull a small model to verify: ollama pull qwen2.5:1.5b"

echo "== llama.cpp (source build with Metal) =="
echo "Run manually (not cloned automatically by this script):"
echo "  git clone https://github.com/ggerganov/llama.cpp"
echo "  cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j"

echo "== Python-side runtime packages (from the repo root) =="
echo "  CMAKE_ARGS=\"-DGGML_METAL=ON\" uv add \"llama-cpp-python[server]\""
echo "  uv add mlx mlx-lm openai"

echo "== Done. Verify with: =="
echo "  uv run python scripts/module_02/smoke_test_runtimes.py"
