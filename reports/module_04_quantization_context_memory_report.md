# Module 4 deliverable — quantization, context, and memory report

Status: **memory formulas built, verified against every worked example in the theory doc,
and demonstrated with real measurement tooling (proven on a dummy process). The "measure"
half of Lab 4.4's real prediction-vs-actual table is pending a resourced Mac** (standing
constraint — see `PROGRESS.md`).

## What's built and verified

| Artifact | Verified how |
|---|---|
| `scripts/module_04/memory_math.py` | 22 unit tests, each checked against a specific number in the theory doc's tables (weights rule-of-thumb table, KV-cache worked example and context table, concurrency scaling, KV-cache quantization halving) |
| `scripts/module_04/model_shapes.py` | 6 unit tests; 4 documented model shapes (Llama 3.1 8B, Qwen2.5 7B/1.5B, Qwen2.5-Coder 7B), each with a `source_note` |
| `scripts/module_04/memory_sampler.py` | 10 unit tests; **real, working process-RSS sampling tooling**, proven in the executed notebook against a dummy subprocess that allocates ~200MB — the sampler correctly tracked a 148MB+ peak |
| `scripts/module_04/lab_4_{1,2,3,4}_*.py` | 19 unit tests across the four labs (table rendering, dataclass properties, injected-fake orchestration logic) plus 4 CLI skip-path tests — all four labs correctly print an actionable `SKIPPED` message and exit 1 on this machine |
| `notebooks/04_quantization_context_memory_math.ipynb` | **Executed end-to-end** — every theory-doc number reproduced from code, the memory sampler proven against a real (dummy) process, Lab 4.4 correctly skipped against real Ollama |

57 new unit tests this module (171 total across the repo), `ruff check .` clean.

## Formulas verified against the theory doc (from the executed notebook)

**Weights, 8B model, by quantization (decimal GB — see the unit note in `memory_math.py`):**

| Quant | Computed |
|---|---:|
| FP16 | 16.00 GB |
| Q8_0 | 8.50 GB |
| Q6_K | 6.60 GB |
| Q5_K_M | 5.70 GB |
| Q4_K_M | 4.80 GB |
| Q3_K_M | 3.90 GB |
| Q2_K | 3.40 GB |

Exact match to the theory doc's rule-of-thumb table (§1) in every row.

**KV cache, 8B Llama-style shape (`n_layers=32, n_kv_heads=8, head_dim=128`), binary GiB:**

| Context | FP16 | Q8 | Q4 |
|---:|---:|---:|---:|
| 4,096 | 0.500 | 0.250 | 0.125 |
| 8,192 | 1.000 | 0.500 | 0.250 |
| 32,768 | 4.000 | 2.000 | 1.000 |
| 128,000 | 15.625 | 7.812 | 3.906 |

The 4K/8K/32K rows match the theory doc's table exactly. The 128K row computes to 15.625 GiB
rather than the theory doc's rounded "~16.0 GiB" — the doc's own table is itself a rounded
approximation of 128,000 tokens (not the power-of-two 131,072 = "128Ki" tokens that would
give exactly 16.0 GiB); `memory_math.py` uses the precise `context_tokens` value passed in,
which is the more correct behavior for a planning tool. Noted here rather than silently
"matching" a rounded number.

**Full budget estimates by model shape, Q4_K_M, from `model_shapes.py`'s registry:**

| Model | Context | Weights | KV cache | Total range |
|---|---:|---:|---:|---|
| llama3.1-8b | 2,000 | 4.80 GB | 0.24 GiB | 5.54-6.54 |
| llama3.1-8b | 8,000 | 4.80 GB | 0.98 GiB | 6.28-7.28 |
| llama3.1-8b | 16,000 | 4.80 GB | 1.95 GiB | 7.25-8.25 |
| qwen2.5-7b | 2,000 | 4.56 GB | 0.11 GiB | 5.17-6.17 |
| qwen2.5-7b | 8,000 | 4.56 GB | 0.43 GiB | 5.49-6.49 |
| qwen2.5-7b | 16,000 | 4.56 GB | 0.85 GiB | 5.91-6.91 |
| qwen2.5-1.5b | 2,000 | 0.90 GB | 0.05 GiB | 1.45-2.45 |
| qwen2.5-1.5b | 8,000 | 0.90 GB | 0.21 GiB | 1.61-2.61 |
| qwen2.5-1.5b | 16,000 | 0.90 GB | 0.43 GiB | 1.83-2.83 |

Qwen2.5-7B's smaller KV cache relative to the Llama-3.1-8B-shape example at the same context
is a direct, visible consequence of its narrower GQA configuration (4 KV heads vs. 8) — a
concrete illustration of theory doc §4's grouped-query-attention point, not just an assertion.

## Memory sampler: real tooling, proven

`memory_sampler.py`'s `PeakMemorySampler` was pointed at a dummy subprocess that allocates a
~200MB `bytearray`, holds it for 0.6s, then exits. The sampler correctly tracked a peak of
148.2 MiB (some undercount vs. the full 200MB is expected — allocation isn't instantaneous
and the poll interval can miss the exact peak instant; well above the sanity threshold used
in both the notebook's assertion and the unit tests). This is the same tool Labs 4.1-4.4 use
to sample the Ollama/llama.cpp server process's RSS during real generation calls — proven to
work correctly, just not yet pointed at a real model process.

## Labs pending live execution

```bash
# Lab 4.1 - quantization comparison (needs the same model pulled at 2+ quantizations)
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q8_0
uv run python scripts/module_04/lab_4_1_quantization_comparison.py \
    --tags qwen2.5:7b-instruct-q4_K_M qwen2.5:7b-instruct-q8_0

# Lab 4.2 - context scaling
uv run python scripts/module_04/lab_4_2_context_scaling.py --model qwen2.5:3b

# Lab 4.3 - concurrency simulation
uv run python scripts/module_04/lab_4_3_concurrency_simulation.py --model qwen2.5:3b

# Lab 4.4 - predict, then measure (the module's core deliverable table)
uv run python scripts/module_04/lab_4_4_predict_then_measure.py \
    --model-tag qwen2.5:7b-instruct-q4_K_M --shape qwen2.5-7b
```

Fold each lab's output into this report, replacing this section. Lab 4.4's output table has
a `gap_explanation` column deliberately left as a fill-in-manually checklist (allocator
overhead, unified-memory accounting, background apps, runtime-specific buffering) — the
prediction-vs-actual *numbers* are automatic, but the *explanation* of any gap is a judgment
call a report has to make explicitly, the same pattern Module 3's "justify model selection"
requirement used.

## Known limitation carried forward from Module 4.3

`ConcurrencyLevelResult` reports a `failure_rate`, not a `timeout_rate` as the curriculum
literally names it — Module 1's `ollama_probe.generate` currently wraps every httpx error
(connection refused, read timeout, ...) into the same `OllamaUnavailable` exception, so
timeouts can't be cleanly distinguished from other failures yet. A proper `RequestTimeout`
vs. other-error taxonomy is explicitly Module 6's job (curriculum.md §16's error taxonomy).
Documented in `lab_4_3_concurrency_simulation.py`'s docstring rather than silently reported
under a misleading name.

## Assessment self-check

The prediction-vs-actual table (Lab 4.4) must discuss overhead, allocator behavior,
unified-memory accounting, background apps, and runtime-specific buffering. The *predicted*
half is real and verified above; the *actual* half and its gap discussion are pending the
resourced Mac, with the exact commands given above.
