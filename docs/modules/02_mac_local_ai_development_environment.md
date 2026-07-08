# Module 2 — Mac Local AI Development Environment

> Phase: Foundation · Bible reference: [curriculum.md §12](../../curriculum.md#12-module-2--mac-local-ai-development-environment)

## Goal

Turn a Mac into a reliable local AI development workstation: the dev tools, package
manager, and three model runtimes (Ollama, llama.cpp/llama-cpp-python, MLX) that every later
module assumes are present and working.

> **Machine note for this repo:** the Mac this course is being authored on has limited disk
> and memory and is **not used to install or run any model runtime or model weights**. This
> module is written, tested (where testable without a runtime), and documented so it is
> correct and ready to execute on a properly resourced Mac. See
> [reports/module_02_environment_report.md](../../reports/module_02_environment_report.md)
> for exactly what has and hasn't been executed here, and why.

## 1. Apple Silicon vs Intel Mac

This matters for local LLM work more than for typical app development, because it changes
which runtimes are worth using at all:

| | Apple Silicon (M-series) | Intel Mac |
|---|---|---|
| Unified memory | Yes — CPU/GPU/Neural Engine share one pool (Module 1 §4) | No — discrete or integrated GPU with separate/limited VRAM behavior |
| MLX support | Native, first-class | Not supported — MLX is Apple Silicon-only |
| Metal acceleration (llama.cpp) | Full support, strong performance | Limited/no Metal GPU acceleration path |
| Ollama | Fully supported, GPU-accelerated via Metal | Supported, CPU-only in practice |
| Practical local-LLM ceiling | Meaningfully higher for a given RAM figure | Lower — expect to lean on the smallest model tiers |

Detect which you have before assuming any runtime's performance characteristics:

```bash
uname -m        # arm64 = Apple Silicon, x86_64 = Intel
sysctl -n machdep.cpu.brand_string
```

## 2. macOS developer tools

Xcode Command Line Tools provide the compilers and build tooling that `cmake`,
`llama-cpp-python`'s build step, and various native Python wheels depend on:

```bash
xcode-select --install
```

## 3. Homebrew

Homebrew is the package manager this course standardizes on for system-level tools (not
Python packages — that's `uv`'s job). Install from <https://brew.sh> if not already present,
then confirm:

```bash
brew --version
```

## 4–5. Python environment management and uv-based project setup

This course pins Python 3.12 and manages the environment with `uv`, not a manually-activated
venv or Conda. `uv` gives fast, reproducible installs driven entirely by `pyproject.toml` and
a committed `uv.lock`. Recommended base tools:

```bash
brew install git make cmake python@3.12 uv jq ripgrep
```

The project root of this repo is already a working `uv` project (`pyproject.toml`,
`uv.lock`, `.python-version` pinned to 3.12) — Lab 2.1 below verifies that, rather than
re-creating it from scratch as the curriculum's generic instructions describe, since this
course repo already *is* that reproducible project.

## 6. Ollama installation

Ollama is the fastest path to local experimentation: it manages model downloads, quantized
variants, and exposes a simple local HTTP API (`http://localhost:11434`).

```bash
brew install ollama
brew services start ollama       # run as a background service
# or: ollama serve                # run in foreground for one session

ollama pull qwen2.5:1.5b          # pull a small model to verify the path end-to-end
ollama run qwen2.5:1.5b "Say hello in five words."
```

Model files are cached under `~/.ollama/models`. This can grow large quickly — a handful of
7B-class quantized models is easily tens of gigabytes; see §10 and §12 below.

## 7. llama.cpp and Metal

llama.cpp is the reference GGUF runtime and gives the most low-level control over
quantization, context size, and Metal GPU offload. Build with Metal support on Apple Silicon:

```bash
brew install cmake
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

The build produces a `llama-server` binary that serves an OpenAI-compatible HTTP API —
this is the piece Module 5 (serving patterns) and Lab 2.3 below depend on.

## 8. llama-cpp-python with server extras

For Python-native integration (rather than shelling out to the `llama-server` binary),
`llama-cpp-python` wraps llama.cpp and can itself run an OpenAI-compatible server, with
Metal acceleration via a build flag:

```bash
CMAKE_ARGS="-DGGML_METAL=ON" uv add llama-cpp-python
# or, for just the server extras:
CMAKE_ARGS="-DGGML_METAL=ON" uv add "llama-cpp-python[server]"
```

Gotcha (from the bible): build/config issues on Mac can happen — the C++ build step is the
most common source of installation failure in this course. If it fails, capture the exact
build error in the environment report rather than silently skipping the runtime.

## 9. MLX and mlx-lm

MLX is Apple's array framework built for unified memory and Apple Silicon; `mlx-lm` builds
LLM inference and fine-tuning on top of it. Apple Silicon only (§1).

```bash
uv add mlx mlx-lm
uv run python -m mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --prompt "Say hello in five words."
```

`mlx-lm` downloads models from Hugging Face on first use (cached under
`~/.cache/huggingface`), not through Ollama's cache.

## 10. Model cache management

Three runtimes in this module use **three different cache locations**, which is a common
source of "where did my disk space go":

| Runtime | Default cache location | Notes |
|---|---|---|
| Ollama | `~/.ollama/models` | Managed by `ollama rm <model>`; not a plain Hugging Face cache |
| llama.cpp / llama-cpp-python | Wherever you point `--model` / `model_path` | You manage these GGUF files directly — no automatic cache |
| MLX / mlx-lm | `~/.cache/huggingface/hub` | Standard Hugging Face cache; shared with any other HF-based tooling |

`scripts/module_02/model_cache.py` (built below) locates and sizes each of these
directories so "disk usage and cleanup" (§12) is a measured fact, not a guess.

## 11. Reproducibility

A local AI dev environment is reproducible when a fresh clone plus a documented set of
commands gets a new machine to the same working state. For this course that means:

- `pyproject.toml` + `uv.lock` fully describe the Python dependency graph — `uv sync`
  reproduces it exactly;
- runtime installation (Ollama, llama.cpp, MLX) is captured as explicit, copyable commands
  in this document and in `scripts/module_02/setup_mac.sh`, not tribal knowledge;
- which model *tags/quantizations* were tested is recorded (this becomes
  `models/MODEL_CATALOG.md` starting Module 3) rather than "whatever I happened to have
  pulled."

## 12. Disk usage and cleanup

Local models are large and multiply across runtimes (§10). Before pulling multiple models
across all three runtimes, budget disk space deliberately:

```bash
du -sh ~/.ollama/models 2>/dev/null
du -sh ~/.cache/huggingface 2>/dev/null
df -h /
```

Cleanup commands:

```bash
ollama rm <model>                          # remove a specific Ollama model
ollama list                                # see what's resident
rm -rf ~/.cache/huggingface/hub/<model-dir>  # remove a specific HF-cached model (be exact)
```

`scripts/module_02/model_cache.py`'s reporting function is the automated version of the
`du -sh` commands above, run consistently across all three cache locations.

## Hands-on labs

### Lab 2.1 — Create a reproducible Python project

**Status: satisfied by this repo itself.** The root of this repo already is the reproducible
project the curriculum's generic instructions describe: `pyproject.toml`, `uv.lock`,
`Makefile`, `README.md`, and `packages/*/tests/` + `scripts/module_NN/tests/` all exist and
are exercised by `make test`. `scripts/module_02/mac_environment_check.py` verifies the
required CLI tools (§4–5) are present and prints their versions.

### Lab 2.2 — Run a model through Ollama

Run `scripts/module_02/smoke_test_ollama.py`. On a machine with Ollama installed and a model
pulled, it lists local models, runs a fixed prompt, and reports token counts, TTFT, and
tokens/sec via the same `ollama_probe.py` helper Module 1 built. On this machine (no Ollama),
it prints the exact install/pull commands and exits with an explicit skip — see the
deliverable report.

### Lab 2.3 — Run a model through the llama-cpp-python server

Run `scripts/module_02/smoke_test_llamacpp_server.py`, which checks whether
`llama_cpp` is importable and a `LLAMACPP_MODEL_PATH` is set, then calls the OpenAI-compatible
server through the `openai` Python client. Skips with instructions otherwise.

### Lab 2.4 — Run a model through MLX

Run `scripts/module_02/smoke_test_mlx.py`, which checks whether `mlx_lm` is importable and an
`MLX_MODEL_ID` is set, then generates text and reports wall-clock latency. Skips with
instructions otherwise.

## Deliverable

`reports/module_02_environment_report.md`, assembled by
`scripts/module_02/smoke_test_runtimes.py`, which runs all three smoke tests plus the
environment/tool check and cache report and renders one combined markdown report.

## Assessment

- [ ] One prompt executed through at least two runtimes — **deferred to the resourced Mac**;
      commands are documented and scripts are ready.
- [x] Model files and caches located and documented (§10, `model_cache.py`).
- [ ] Benchmark metrics captured — deferred with the above.
- [x] Failure notes recorded for any runtime that does not work on the current Mac — this
      machine deliberately has none installed; recorded as a machine constraint, not a
      runtime failure.
- [x] Environment can be recreated from README — `README.md`'s Module 2 section gives the
      exact commands.
