#!/usr/bin/env bash
# Module 5 Lab 1 — start Ollama through a repeatable command.
#
# Documentation-as-script, same pattern as scripts/module_02/setup_mac.sh:
# reviewed and understood before running, NOT executed as part of this
# repo's build/test process, and must never be run on a machine that
# shouldn't have a model runtime installed (see
# docs/modules/02_mac_local_ai_development_environment.md's machine
# constraint, which applies to every module in this repo).
#
# Usage (on a resourced Mac):
#   bash scripts/module_05/serve_ollama.sh [model_tag]
#
# Idempotent: if Ollama is already serving on port 11434, this reports that
# and exits 0 instead of starting a second instance.

set -euo pipefail

MODEL_TAG="${1:-qwen2.5:3b}"
BASE_URL="http://localhost:11434"

if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already serving at ${BASE_URL}."
else
  echo "Starting Ollama..."
  # `brew services start ollama` runs it as a managed background service;
  # `ollama serve` (foreground) is the alternative for a single terminal session.
  brew services start ollama

  echo "Waiting for Ollama to become reachable..."
  for _ in $(seq 1 30); do
    if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
    echo "Ollama did not become reachable within 30s. Check \`brew services list\` and logs." >&2
    exit 1
  fi
  echo "Ollama is up."
fi

echo "Ensuring ${MODEL_TAG} is pulled..."
ollama pull "${MODEL_TAG}"

echo "Warming the model (first request pays load cost - see warmup_experiment.py)..."
curl -fsS "${BASE_URL}/api/generate" \
  -d "{\"model\": \"${MODEL_TAG}\", \"prompt\": \"Say hello.\", \"stream\": false}" \
  >/dev/null

echo "Ready. Verify with:"
echo "  uv run python scripts/module_05/feature_matrix.py"
echo "  uv run python scripts/module_05/ollama_metadata.py --model ${MODEL_TAG}"
