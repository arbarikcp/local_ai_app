# Module 1 deliverable — local LLM observations

Status: **infrastructure complete, empirical labs pending a local runtime.**

## Environment at time of writing

- Date: 2026-07-08
- Machine: macOS (Darwin 22.1.0)
- Python: 3.13.5 system / 3.12 pinned for this project via `uv`
- **Ollama: not installed** — `is_ollama_available()` returns `False`; verified via
  `scripts/module_01/tests/test_ollama_probe.py::test_is_ollama_available_returns_false_when_unreachable`
  and by direct run of the notebook (see `notebooks/01_local_llm_basics.ipynb`, cell output).
- llama.cpp / llama-cpp-python / MLX: not yet verified — that verification belongs to
  Module 2 (`scripts/smoke_test_*`), not this module.

Per the course's own honesty rule (bible §4.1: "Do not promise that a model 'runs on 8 GB'
unless the exact quantization, context, and runtime have been tested"), the three labs below
are **not marked complete** and contain no fabricated numbers. What follows is what has
actually been built and verified, plus the exact commands to complete each lab once a
runtime is installed (Module 2).

## What's built and verified

| Artifact | Verified how |
|---|---|
| `scripts/module_01/ollama_probe.py` | Unit tested (`test_ollama_probe.py`, 6 tests) — TTFT/tokens-per-second derivation logic and availability check |
| `scripts/module_01/token_estimate.py` | Unit tested (`test_token_estimate.py`, 6 tests) — heuristic estimate, inverse word count, exact-tokenizer failure path |
| `scripts/module_01/lab_1_1_multi_model_run.py` | Report-formatting logic unit tested (`test_report_formatting.py`); executed live in notebook against unreachable Ollama, confirmed clean skip with no fabricated output |
| `scripts/module_01/lab_1_2_long_prompt_stress_test.py` | Prompt-builder logic unit tested (`test_lab_1_2_prompt_builder.py`, 3 tests); report-formatting tested |
| `scripts/module_01/lab_1_3_small_model_failure_analysis.py` | CLI runs against live Ollama only; manual-annotation format verified by inspection (not yet executed against a real model) |
| `notebooks/01_local_llm_basics.ipynb` | **Executed end-to-end** via `jupyter nbconvert --execute` — memory-math cells produce real computed numbers; Ollama-dependent cells correctly print skip notices |

All 19 unit tests pass (`uv run pytest -q`) and `ruff check .` is clean.

## Memory math derived and verified (from the notebook, real computed values)

For a 7B-parameter, 28-layer, 4-KV-head, 128-head-dim model (Qwen2.5-7B-class shape):

**Weights by quantization:**

| Quant | Computed weight footprint |
|---|---:|
| FP16 | 13.04 GiB |
| Q8_0 | 6.93 GiB |
| Q6_K | 5.38 GiB |
| Q5_K_M | 4.64 GiB |
| Q4_K_M | 3.91 GiB |

**KV cache by context length (single sequence, FP16 cache):**

| Context | Computed KV cache |
|---:|---:|
| 4,096 | 0.219 GiB |
| 8,192 | 0.438 GiB |
| 32,768 | 1.750 GiB |
| 128,000 | 6.836 GiB |

**KV cache by concurrency (8K context, FP16 cache):**

| Concurrent requests | Computed KV cache |
|---:|---:|
| 1 | 0.438 GiB |
| 2 | 0.875 GiB |
| 4 | 1.750 GiB |
| 8 | 3.500 GiB |

These are formula-derived (see `docs/modules/01_local_llm_systems_thinking.md` §3, §10), not
measured against a running process — they are the "predict" half of Module 4's
"predict, then measure" lab, done early here to motivate the memory model. Module 4 will
compare these predictions against actual measured peak memory.

## Labs pending live execution

Run once Ollama is installed and at least `qwen2.5:1.5b`, `qwen2.5:3b`, `qwen2.5:7b` are
pulled (Module 2 covers install):

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b   # if RAM tier allows

uv run python scripts/module_01/lab_1_1_multi_model_run.py > /tmp/lab_1_1_raw.md
uv run python scripts/module_01/lab_1_2_long_prompt_stress_test.py --model qwen2.5:3b > /tmp/lab_1_2_raw.md
uv run python scripts/module_01/lab_1_3_small_model_failure_analysis.py --model qwen2.5:1.5b > /tmp/lab_1_3_raw.md
```

Then:

1. Append the Lab 1.1 table below, with quality notes added by hand after reading each
   model's actual answer.
2. Append the Lab 1.2 table below, adding manually-observed peak memory per row (Activity
   Monitor or `ps -o rss -p <ollama_pid>` sampled during each run — the script does not
   sample this automatically, see the script's docstring for why).
3. Append the Lab 1.3 output below, with the "Failure mode observed" line filled in by hand
   for each of the five tasks.

### Lab 1.1 — multi-model comparison

_Pending live run — see command above._

### Lab 1.2 — long prompt stress test

_Pending live run — see command above._

### Lab 1.3 — small model failure analysis

_Pending live run — see command above._

## Assessment self-check (can be answered now, independent of live runs)

1. **Why does context length affect memory?** Every token in context adds one key vector and
   one value vector per layer to the KV cache (`kv_bytes ≈ 2 × n_layers × n_kv_heads ×
   head_dim × context_tokens × bytes_per_element × concurrent_sequences`). Weights are fixed
   for a given quantization; KV cache is not — it scales linearly with context and
   concurrency, which is why a model that loads fine can still exceed available memory once
   the prompt grows or multiple requests arrive concurrently.
2. **Why can Q4 and Q8 behave differently?** They differ in both bytes-per-parameter (memory
   footprint) and in numerical precision retained per weight, which changes generation
   quality, and can change speed on memory-bandwidth-bound decoding since fewer bytes per
   weight means less data to move per token.
3. **Why can a local model be private but insecure?** Local execution only guarantees the
   prompt/response for a given call doesn't leave the machine over the network. It says
   nothing about prompt injection from retrieved/tool content, unsafe tool execution,
   insecure storage of logs/history/vector indexes, or missing access control — all separate
   concerns covered in Module 22.
4. **Why do small models need stricter application architecture?** They are more prone to
   confident fabrication, instruction drift over long/complex prompts, format collapse under
   structured-output tasks, and weak multi-step reasoning (§11 of the theory doc) — so the
   application (prompts, schemas, retrieval, validation, retries) has to compensate for
   capability the model itself doesn't reliably provide.
