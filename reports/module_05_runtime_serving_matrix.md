# Module 5 deliverable — runtime serving matrix

Status: **all serving-behavior parsing/orchestration logic built and unit-tested; the
feature matrix populated from documentation. Real per-runtime measurement (flipping entries
from "documented" to "measured") is pending the resourced 32GB Mac** (standing constraint —
see `PROGRESS.md`).

## What's built and verified

| Artifact | Verified how |
|---|---|
| `scripts/module_05/serve_ollama.sh`, `serve_llamacpp.sh` | Syntax-checked (`bash -n`); reviewed, not executed (would start a runtime) |
| `scripts/module_05/ollama_streaming.py` | 15 unit tests against fixture NDJSON lines shaped like real Ollama output — parsing, chunk accumulation, real TTFT-from-first-chunk, tokens/sec from the final `done` chunk |
| `scripts/module_05/ollama_metadata.py` | 9 unit tests against a fixture `/api/show` response shaped like a real Qwen2.5 model — including the dynamic `<family>.context_length` key lookup |
| `scripts/module_05/warmup_experiment.py` | 11 unit tests — cold/warm orchestration, mean/speedup statistics, None-handling when a call fails |
| `scripts/module_05/feature_matrix.py` | 9 unit tests — matrix completeness, markdown rendering, the `documented`→`measured` verification-status flip |
| `scripts/module_05/llamacpp_openai_streaming.py` | 7 unit tests — chat-stream accumulation, CLI skip path |
| `scripts/module_05/run_mlx_generate.py` | 8 unit tests — **including a regression test for a real bug this module caught mid-build** (see below) |
| `notebooks/05_serving_local_models.ipynb` | **Executed end-to-end** — parsers proven against fixtures, feature matrix rendered, warmup statistics proven with a fake TTFT function, real-runtime cells correctly skip |

59 new unit tests this module (230 total across the repo), `ruff check .` clean.

## A real bug caught during this module's own build

While writing `run_mlx_generate.py`'s `summary_to_markdown`, an f-string ternary was
structured as:

```python
return (
    f"...line 1..."
    f"...line 2..."
    f"...{value}...\n"
    if condition
    else f"...\n"
)
```

Adjacent string literals concatenate *before* the ternary applies in Python, so this
silently collapsed the **entire** multi-line report down to just the `else` branch's single
line whenever `condition` was false — every other line (load time, cold/warm generate,
warmup speedup) would have vanished. Caught by writing
`test_does_not_collapse_when_stream_total_is_none` before considering the function done, not
by manual inspection. Fixed by computing the conditional piece as its own string first. Left
the regression test in place with a comment explaining what it guards against, per this
module's "no shortcuts" instruction — this is exactly the kind of bug unit tests exist to
catch, and it did.

## Runtime feature matrix (from the executed notebook)

| Runtime | Structured output | Grammar | Token counting | Streaming | Cancellation | Usage reporting |
|---|---|---|---|---|---|---|
| ollama | yes (documented) | no (documented) | partial (documented) | yes (documented) | yes (documented) | yes (documented) |
| llama.cpp (llama-server) | yes (documented) | yes (documented) | yes (documented) | yes (documented) | yes (documented) | yes (documented) |
| llama-cpp-python[server] | yes (documented) | yes (documented) | yes (documented) | yes (documented) | partial (documented) | yes (documented) |
| MLX / mlx-lm | no (documented) | no (documented) | partial (documented) | yes (documented) | n/a (documented) | partial (documented) |

All 24 feature entries (4 runtimes × 6 features) are currently `documented`, sourced from
public runtime documentation as of this course's authoring — **none measured** on this
machine. Per-feature notes (grammar support, token-counting endpoints, etc.) are in
`feature_matrix.py`'s `notes_appendix()`. Runtime feature support changes across versions;
treat every "yes" here as a claim to verify, exactly like `models/MODEL_CATALOG.md` requires
for model claims (Module 3).

## Machine profile

This machine **is** Apple Silicon (arm64) — confirmed by `run_mlx_generate.is_apple_silicon()`
returning `True` in the executed notebook — but `mlx_lm` is not installed (by design, per the
machine constraint), so the MLX lab correctly prints a skip message rather than a false
"not Apple Silicon" one. Worth noting for the resourced Mac: if it's also Apple Silicon
(need to confirm — the 32GB figure alone doesn't tell us the chip), the MLX labs become
directly runnable; if it's Intel, Lab 6 (MLX) is a hard skip regardless of RAM, and only the
Ollama/llama.cpp labs apply.

## Labs pending live execution

```bash
# Lab 1 - start each runtime
bash scripts/module_05/serve_ollama.sh qwen2.5:3b
bash scripts/module_05/serve_llamacpp.sh python /path/to/model.gguf   # or: native

# Lab 2/5/6 - native API, warmup, metadata
uv run python scripts/module_05/ollama_metadata.py --model qwen2.5:3b
# (warmup_experiment.py and ollama_streaming.py are used as library calls - see the notebook's
# real-runtime cell for the wiring, or call stream_generate/run_warmup_experiment directly)

# Lab 3/4 - OpenAI-compatible streaming
uv run python scripts/module_05/llamacpp_openai_streaming.py --base-url http://localhost:8080/v1

# MLX
uv run python scripts/module_05/run_mlx_generate.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

After running, update `feature_matrix.py`'s `KNOWN_FEATURE_MATRIX` entries from
`verified=False` to `verified=True` with real behavioral notes for whichever features were
actually exercised, and paste real output into this report replacing this section.

## Assessment self-check

> "The matrix must document feature support and observed behavior for each runtime."

- **Feature support, documented**: done — all 24 entries populated from public
  documentation with per-feature notes.
- **Observed behavior**: pending the resourced Mac. The parsing/measurement code that will
  produce real observations (`ollama_streaming.py`'s real TTFT, `warmup_experiment.py`'s
  cold/warm comparison, `ollama_metadata.py`'s live `/api/show` probe) is built and
  unit-tested now, so completing this is a matter of running it, not building it.

## Gotchas encountered (beyond the ternary bug above)

- `ollama_metadata.py`'s context-length lookup has to search `model_info` keys dynamically
  (`f"{family}.context_length"`) rather than a fixed key, because the family-prefixed key
  name depends on the model's architecture string — confirmed against a realistic fixture,
  not assumed.
- `lab 4.3`'s (Module 4) `failure_rate`-vs-`timeout_rate` naming gap, noted in Module 4's
  report, is still open — this module's `ollama_streaming.py` inherits the same limitation
  (a cancelled/timed-out stream and a connection-refused stream both surface as generic
  httpx exceptions). Still deferred to Module 6's error taxonomy, not fixed ad hoc here.
