# Module 4 — Quantization, Context, and Memory Math

> Phase: Foundation · Bible reference: [curriculum.md §14](../../curriculum.md#14-module-4--quantization-context-and-memory-math)

## Goal

Understand why models that "fit" may still fail under production use. Module 1 previewed the
weights + KV-cache formula; this module makes it exact, adds the full memory budget,
and — critically — turns it into a **predict, then measure** discipline rather than a
one-time calculation.

> **Machine note:** this repo is built on a Mac that must never run a model runtime
> ([[project-local-ai-app-curriculum]] constraint). Every formula and tool in this module is
> unit-tested against known values; the "measure" half of "predict, then measure" is pending
> a resourced Mac — see `reports/module_04_quantization_context_memory_report.md`.

## 1. Quantization formats: FP16 down to Q2

| Quant | Approx bits/param | Rule of thumb for an 8B model |
|---|---:|---|
| FP16 | 16.0 | ~16.0 GB |
| Q8_0 | ~8.5 | ~8.5 GB |
| Q6_K | ~6.6 | ~6.6 GB |
| Q5_K_M | ~5.7 | ~5.7 GB |
| Q4_K_M | ~4.8 | ~4.8 GB |
| Q3_K_M | ~3.9 | ~3.9 GB |
| Q2_K | ~3.4 | ~3.4 GB, but quality is often unacceptable |

GGUF "K-quants" (`Q4_K_M`, `Q5_K_M`, ...) store per-block scale factors, so the bits/param
figure above is an **effective average**, not an exact per-weight value — two models
quantized to "Q4_K_M" can differ slightly in actual size depending on block layout. Treat
the table as planning-grade, and get the exact file size from the runtime/file itself before
finalizing a memory budget.

## 2. GGUF quantization names, decoded

The naming convention (`Q<bits>_<variant>_<size>`) encodes real information:

- **`Q<N>`** — nominal bits per weight (4, 5, 6, 8...).
- **`_0` / `_1`** — legacy simple quantization (single scale per block, `_1` adds a min offset).
- **`_K`** — "k-quant": mixed-precision blocks that spend more bits on more sensitive weights.
- **`_S` / `_M` / `_L`** — small/medium/large variants of a k-quant, trading size for quality
  within the same nominal bit-width.

`Q4_K_M` is the course's practical default starting point for the 16 GB tier specifically
because it's usually the best quality-per-byte point on that curve — but "usually" means
"benchmark it for your task" (Module 3), not "assume it."

## 3. Quality/performance trade-offs

Lower quantization is the single biggest lever for making a model fit and run acceptably on
constrained RAM — and also the easiest way to silently degrade quality below what a task
needs. Two things move together as bits/param drops:

- **Memory footprint** drops roughly linearly with bits/param (the weights formula, §5).
- **Generation speed** often *improves* too, since decoding is memory-bandwidth-bound
  (Module 1 §8) — fewer bytes per weight means less data to move per generated token.
- **Quality** degrades, usually gracefully down to Q4-ish and then increasingly sharply
  below that — which is why Q2/Q3 need explicit benchmarking (Module 3's harness) before
  ever shipping, not just "it loaded and produced text."

## 4. KV cache, precisely

Revisiting Module 1 §3 with the exact formula this module operationalizes:

```text
kv_bytes ≈ 2                                  # K and V
         × n_layers
         × n_kv_heads × head_dim
         × context_tokens
         × bytes_per_element(kv_quant)
         × concurrent_sequences
```

Use `n_kv_heads`, **not** the full attention head count — grouped-query attention (GQA)
means many modern models share key/value projections across groups of query heads, so the
KV cache is much smaller than a naive "per attention head" calculation would suggest.

Worked example, 8B-class Llama-style model (`n_layers=32`, `n_kv_heads=8`, `head_dim=128`):

```text
KV width per layer = 8 × 128 = 1024
Per-token elements = 2 × 32 × 1024 = 65,536 elements
At FP16 (2 bytes/element) = 65,536 × 2 = 128 KiB/token
```

| Context | KV cache at FP16 | KV cache at Q8 | KV cache at Q4 |
|---:|---:|---:|---:|
| 4K | ~0.5 GiB | ~0.25 GiB | ~0.125 GiB |
| 8K | ~1.0 GiB | ~0.5 GiB | ~0.25 GiB |
| 32K | ~4.0 GiB | ~2.0 GiB | ~1.0 GiB |
| 128K | ~16.0 GiB | ~8.0 GiB | ~4.0 GiB |

**The punchline of this module:** an 8B model whose weights fit in roughly 5 GB can still
exceed a 16 GB machine purely from context and concurrency. An advertised 128K context does
not mean a *usable* 128K context on an 8 GB Mac — it means the architecture supports it,
memory permitting.

## 5. Context length and memory — the full budget

```text
total ≈ weights + kv_cache + runtime_overhead + compute_buffers + app_memory + OS_memory
```

Worked example — 8B Q4_K_M at 8K context, single sequence:

```text
4.8 GB weights
+ ~1.0 GiB FP16 KV cache
+ ~0.5-1.5 GB runtime overhead and compute buffers
+ app memory
+ macOS resident memory
```

This is why the 8 GB course tier should prefer 1B–4B models and strict context budgets — the
student should be able to *derive* this number, not memorize a rule of thumb.

## 6. Prompt compression

When the KV-cache term dominates the budget, shortening the prompt is a direct memory lever,
not just a latency one. Options in increasing order of complexity: trim redundant
instructions/examples, summarize long conversation history (Module 8.5's job long-term),
retrieve only the most relevant chunks instead of dumping a whole document (Module 11's
job), and — as a last resort — use a smaller model that needs less "convincing" per prompt
token.

## 7. Batch size and 8. Concurrent requests

Both multiply the KV-cache term directly (`concurrent_sequences` in the formula above).
Module 6.5 covers scheduling/queueing in depth; this module's job is just to make the memory
consequence concrete:

| Concurrent requests | KV cache at 8K context, FP16 (8B-class shape above) |
|---:|---:|
| 1 | ~1.0 GiB |
| 2 | ~2.0 GiB |
| 4 | ~4.0 GiB |
| 8 | ~8.0 GiB |

## 9. Apple unified memory

Revisit Module 1 §4: there is no separate VRAM safety margin on Apple Silicon. Every term in
the budget formula — weights, KV cache, runtime overhead, the rest of the OS — draws from
the *same* physical pool. A memory budget that "fits" in isolation can still fail in
practice because Chrome, Slack, and the OS itself are drawing from that same pool
concurrently.

## 10. Runtime overhead

Beyond weights and KV cache, expect measurable overhead from: compute/activation buffers for
the forward pass, the runtime's own process memory (Python interpreter, server framework),
model-loading scratch space (sometimes momentarily higher than steady-state), and
allocator fragmentation/rounding. This course budgets **0.5–1.5 GB** as a planning-grade
placeholder for this term (see the worked example, §5) — Lab 4.4 measures the real gap.

### KV-cache quantization is a first-class lever

Many runtimes can quantize the KV cache independently of the model weights (llama.cpp-style
runtimes expose cache type controls; Ollama exposes KV-cache behavior through runtime
configuration). Halving KV precision often costs less quality than lowering model weight
precision, and can buy much more usable context — this is frequently a better lever than
"just use a smaller model."

### Reranker and embedder memory contention

A RAG request often loads more than one model at once: `generator + embedder + optional
cross-encoder reranker`. A cross-encoder reranker can become a *third* resident model during
a single RAG query. On 8–16 GB Macs this can be the difference between a reliable pipeline
and swap/thrash — run embedders, rerankers, and generators sequentially unless measurement
proves simultaneous residency is safe (Module 12 revisits this in the RAG pipeline
architecture itself).

### The honesty rule this module exists to teach

Wrong: *"This 7B model fits on 8 GB."*

Better: *"This 7B Q4 model loaded on an 8 GB Mac with 4K context and no other heavy apps, but
was not reliable under 8K context or concurrent requests."*

Every claim in this course's reports is written the second way.

## Hands-on labs

All four labs need a live model runtime and are built here in skip-safe form (see Module 1's
honesty rule) — real execution is pending the resourced Mac.

- **Lab 4.1 — Quantization comparison**: same task across Q4/Q5/Q8 (where available) of the
  same model; capture quality, latency, memory, invalid-schema rate, hallucination notes.
  `scripts/module_04/lab_4_1_quantization_comparison.py`.
- **Lab 4.2 — Context scaling**: same model at 2K/4K/8K/16K context, with memory sampled
  throughout (not just latency, unlike Module 1's Lab 1.2 which this extends).
  `scripts/module_04/lab_4_2_context_scaling.py`.
- **Lab 4.3 — Concurrency simulation**: 1/2/4/8 concurrent requests; measure queue wait,
  response latency, timeout rate, memory pressure, thermal-throttling symptoms.
  `scripts/module_04/lab_4_3_concurrency_simulation.py`.
- **Lab 4.4 — Predict, then measure**: compute predicted peak memory (via
  `memory_math.py` + `model_shapes.py`) at 2K/8K/16K context for a chosen model, then measure
  actual peak memory (via `memory_sampler.py`, which is real, working, unit-tested process
  RSS-sampling tooling — proven against a dummy subprocess in this module's notebook since
  there's no model process to sample here). `scripts/module_04/lab_4_4_predict_then_measure.py`.

## Deliverable

```text
scripts/module_04/
  memory_math.py         # weights_bytes, kv_cache_bytes, full budget estimate
  model_shapes.py         # known model architecture shapes (n_layers, n_kv_heads, head_dim, n_params)
  memory_sampler.py       # process RSS peak sampler, macOS `ps`-based
  lab_4_1_quantization_comparison.py
  lab_4_2_context_scaling.py
  lab_4_3_concurrency_simulation.py
  lab_4_4_predict_then_measure.py
reports/module_04_quantization_context_memory_report.md
```

Curriculum's literal deliverable path is `reports/quantization_context_memory_report.md`;
this build uses the repo-wide `reports/module_NN_*.md` naming convention from Modules 1–3
instead, for consistency.

## Assessment

The prediction-vs-actual table (Lab 4.4) must discuss overhead, allocator behavior,
unified-memory accounting, background apps, and runtime-specific buffering. On this machine,
the *predicted* half is real (computed by `memory_math.py`, unit-tested against the worked
examples above); the *actual* half is pending the resourced Mac — see the deliverable report.
