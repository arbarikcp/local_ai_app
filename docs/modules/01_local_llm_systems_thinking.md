# Module 1 — Local LLM Systems Thinking

> Phase: Foundation · Bible reference: [curriculum.md §11](../../curriculum.md#11-module-1--local-llm-systems-thinking)

## Goal

Before writing a single line of application code, build the mental model that everything
later in this course depends on: what a local LLM actually *is* at runtime, what it costs
in memory and time, and why that cost structure forces different application-design
decisions than calling a hosted API.

If this module is skipped, later modules will feel like a grab-bag of framework tutorials.
With this module internalized, RAG, agents, structured output, and optimization all read as
consequences of the same handful of constraints: **memory is finite, context is expensive,
and small models are unreliable in specific, predictable ways.**

## 1. What a local LLM is, operationally

A local LLM is not "a chatbot on your laptop." Operationally it is:

- a large, fixed set of numeric **weights** loaded into memory (RAM, or Apple's unified
  memory shared between CPU/GPU/Neural Engine);
- a **tokenizer** that turns text into integers and back;
- a **forward-pass loop** that, given a sequence of tokens, produces a probability
  distribution over the next token, samples one, appends it, and repeats;
- a growing **KV cache** — intermediate attention state — that lets the model avoid
  recomputing the whole sequence on every new token.

Everything downstream — latency, memory ceilings, context limits, batching behavior — comes
directly from these four facts.

## 2. Local inference vs hosted inference: what actually changes

| Aspect | Hosted API (e.g. a cloud LLM endpoint) | Local inference |
|---|---|---|
| Compute | Elastic, provider-managed, effectively unbounded per request | Fixed to the Mac you're running on |
| Memory | Not your problem | **Your hard constraint** — the topic of this whole module |
| Model quality ceiling | Frontier-scale models available | Bounded by what fits in 8–24 GB |
| Latency profile | Network round-trip dominates for short prompts | Local compute dominates; no network hop, but weaker hardware |
| Concurrency | Provider handles fan-out | You are the capacity planner (see Module 6.5) |
| Cost model | Per-token billing | Sunk hardware cost, "free" marginal tokens, but throughput-limited |
| Privacy | Data leaves the machine | Data stays local — necessary but not sufficient for security (see Module 22) |
| Reliability of small-task behavior | Even "small" hosted models are usually strong | Local small models (1B–4B) fail in specific, learnable ways |

The practical consequence: **local LLM engineering is systems engineering under a hard,
visible resource ceiling.** You cannot paper over a bad architecture by throwing more cloud
GPUs at it — the RAM in front of you is the RAM you have.

## 3. Parameters, weights, activations, KV cache

- **Parameters / weights**: the trained numbers that define the model. Their in-memory
  size is `n_params × bytes_per_param`, and `bytes_per_param` is set by the **quantization**
  level (Module 4 covers this precisely). This is mostly a *fixed* cost for a given model +
  quantization — it does not grow while you use the model.
- **Activations**: the intermediate values computed during a single forward pass. These are
  transient and roughly proportional to batch size and sequence length processed in that
  step, not to the full context so far.
- **KV cache**: for every token already in the context, every layer stores a key vector and
  a value vector so that future tokens can attend to it without recomputing it. This is the
  part that **grows with context length** and is the single most misunderstood source of
  local-LLM memory blowups. A model that loads fine can still fail or crawl once the prompt
  gets long — not because the weights changed, but because the KV cache did.

The rule of thumb to internalize now (derived with real numbers in Module 4):

```text
total_memory ≈ model_weights (~fixed) + KV_cache (grows with context × concurrency)
             + runtime_overhead + compute_buffers + app_memory + OS_memory
```

## 4. RAM vs VRAM vs Apple unified memory

On a traditional GPU machine, system RAM and GPU VRAM are separate pools, and moving data
between them costs time. Apple Silicon Macs use **unified memory**: CPU, GPU, and Neural
Engine share one physical memory pool. This is why a MacBook with 16 GB unified memory can
be a genuinely useful local-LLM machine despite having no discrete GPU — but it also means
the model competes directly with the OS, your browser, and every other open app for the
*same* memory, with no separate VRAM safety margin. "Close your other apps" is not a joke in
this course, it's an operating constraint.

## 5. Tokenization

Models don't see characters or words; they see integer token IDs from a model-specific
vocabulary (commonly 32k–150k+ tokens, via a BPE- or SentencePiece-style tokenizer). Two
different model families almost never share a tokenizer, and token counts for the *same*
text can differ meaningfully between them. This matters immediately and practically:

> **Do not use `tiktoken` (OpenAI's tokenizer) to budget prompts for Llama, Qwen, Gemma,
> Phi, Mistral, or GGUF models.** It is the wrong tokenizer and can be materially wrong.

The correct way to count tokens for a local model, in priority order:

1. Use the runtime's own tokenize endpoint or response metadata (Ollama returns
   `prompt_eval_count`/`eval_count`; OpenAI-compatible servers usually return a `usage`
   block; llama.cpp server exposes a `/tokenize` endpoint).
2. Load the model's actual tokenizer (`transformers.AutoTokenizer.from_pretrained(...)`, or
   the tokenizer embedded in a GGUF file) for pre-flight budgeting before a call is made.
3. Never estimate with a generic word-count heuristic for anything that has to fit a hard
   context budget — heuristics are fine for rough intuition, not for admission control.

Also: **always count tokens on the fully-rendered prompt**, after the chat template has
added role markers, special tokens, system/user/assistant separators, and any tool-call
wrapper — not on the raw string the developer wrote. The rendered prompt is what the model
actually consumes.

## 6. Context window

The context window is the maximum number of tokens (prompt + generation) the model can
attend to at once. Three things to hold in tension:

- A larger advertised context window (e.g. "128K context") is a *capability claim about the
  architecture*, not a promise that your Mac can afford to use all of it — the KV cache for
  128K tokens at FP16 can be tens of gigabytes (worked out precisely in Module 4).
- Context is shared between the prompt you send and the tokens the model generates in
  reply — a full context window means the model cannot continue.
- Most runtimes truncate or error when context is exceeded; behavior differs by runtime and
  must be verified, not assumed (Lab 1.2).

## 7. Prompt tokens vs generated tokens

These are billed/measured separately in almost every local runtime's usage metadata, and
they behave differently performance-wise:

- **Prompt tokens** are processed in a (mostly) parallelizable pass — "prompt evaluation" —
  which is why prompt processing is usually much faster per-token than generation.
- **Generated tokens** are produced one at a time, autoregressively — each new token
  requires a full model forward pass conditioned on everything so far. This sequential
  dependency is *why* generation is the slow part and why tokens/sec for generation is the
  headline performance number, not prompt tokens/sec.

## 8. Time to first token (TTFT) and tokens per second (TPS)

- **TTFT**: wall-clock time from request submission to the first generated token appearing.
  Dominated by prompt evaluation time (which scales with prompt length) plus model
  load/warmup if the model wasn't already resident. This is the number that determines
  whether a UI feels "instant" or "stuck."
- **TPS (tokens/sec)**: steady-state generation speed once decoding is underway. Dominated
  by memory bandwidth for a given model size and quantization on a given chip — not
  primarily by raw FLOPs, because autoregressive decoding is a memory-bandwidth-bound
  workload (every generated token re-reads the whole weight set and the growing KV cache
  from memory).

Both must be measured per model, per quantization, per runtime, per context length — never
assumed from a spec sheet.

## 9. Latency vs throughput

- **Latency**: time for one request to complete (TTFT + generation time).
- **Throughput**: total useful work done per unit time across possibly-concurrent requests.

On a single unified-memory Mac serving one user, these are nearly the same thing. Once you
introduce concurrency (Module 6.5), they diverge: batching multiple requests can *improve*
aggregate throughput while *worsening* the latency (and especially the p95 latency) any
individual user experiences. Treating "it's fast" as a single number is a category error
this course will keep correcting.

## 10. Quantization, in one paragraph (full treatment in Module 4)

Quantization reduces the number of bits used to represent each weight (FP16 → Q8 → Q6 → Q5
→ Q4 → Q3 → Q2), trading model quality for memory footprint and often for speed. Lower
quantization is usually the single biggest lever for making a model fit and run acceptably
on constrained RAM — and also the easiest way to silently degrade quality below what a task
needs. Module 4 gives exact bytes-per-parameter math; for now, hold onto: **quantization is
a deliberate, measured trade-off, not a free lunch.**

## 11. Why small models hallucinate differently

"Hallucination" is not one failure mode. For small local models (roughly 1B–4B parameters)
specific patterns show up repeatedly and are worth naming so you recognize them in Lab 1.3:

- **Confident fabrication under-constrained tasks**: asked an open question, a small model
  will produce fluent, plausible, and sometimes entirely wrong content, with no hedging.
- **Instruction drift over long or multi-part prompts**: small models are far more likely to
  silently drop a rule (e.g. "return only JSON") the further it is from the end of the
  prompt or the more competing instructions surround it.
- **Format collapse under complexity**: asked for structured output with several nested
  fields, small models revert to prose, add commentary, or wrap JSON in markdown fences more
  often than larger models.
- **Weak multi-step reasoning**: chained reasoning (arithmetic, multi-hop lookups, planning)
  degrades faster with model size than single-step tasks do.
- **Over-trust of surface pattern matching**: small models are more likely to answer from a
  memorized-looking pattern than to actually condition on the specific input in front of
  them — which is exactly why retrieval-augmented grounding matters more for them, not less.

None of this means small models are unusable. It means the *application* — prompts,
schemas, retrieval, validation, retries — has to compensate for what the model alone won't
reliably do. That is the throughline of the entire course.

## 12. Why RAG matters more for small local models

A frontier hosted model has broad memorized knowledge and stronger instruction-following, so
it can sometimes "get away with" thin grounding. A small local model has less memorized
knowledge, is more prone to instruction drift, and is exactly the case where **retrieval
quality substitutes for missing model capability.** This is why RAG appears early and
heavily in this course (Modules 11–13) rather than as an advanced add-on: for the model
classes this course targets, RAG is often not optional polish, it's the mechanism that makes
correctness possible at all.

## 13. Local privacy is necessary but not sufficient

Running a model locally means the prompt and generated text never leave the machine over the
network for that inference call — a real and valuable privacy property. It does **not**
mean the application is secure. Locally-run systems still need to worry about: prompt
injection from untrusted retrieved/tool content, unsafe tool execution, data exfiltration
through generated output (e.g., a model persuaded to embed secrets in a response that is
later logged or displayed insecurely), insecure storage of conversation history or vector
indexes, and insufficient access control on retrieval or tools. Module 22 treats this in
depth. For now: **"it's local" is not a security architecture.**

## Mental model summary

```text
Local LLM request cost ≈ fixed model weights (by quantization)
                        + KV cache (grows with context × concurrent sequences)
                        + runtime/compute overhead
                        + everything else running on the Mac

TTFT      ≈ f(prompt length, model residency/warmup)
TPS       ≈ f(model size, quantization, memory bandwidth of the chip)
Reliability of small models ≈ f(task complexity, prompt discipline, retrieval grounding,
                                 output validation) — NOT purely f(model choice)
```

## Hands-on labs

See [scripts/module_01/](../../scripts/module_01/) for runnable lab scripts and
[notebooks/01_local_llm_basics.ipynb](../../notebooks/01_local_llm_basics.ipynb) for the
guided walkthrough. Labs mirror the bible exactly:

- **Lab 1.1** — Run the same prompt across model sizes (1B/3B/7-8B/12-14B where RAM allows)
  and record model, quantization, runtime, prompt/output tokens, TTFT, tokens/sec, peak
  memory, and quality notes.
- **Lab 1.2** — Long-prompt stress test at 500 / 2,000 / 4,000 / 8,000 / 16,000 tokens,
  observing latency growth, memory growth, answer degradation, truncation, and errors.
- **Lab 1.3** — Small-model failure analysis: give a 1B/3B model strict JSON extraction,
  multi-step reasoning, citation-required answering, tool-argument generation, and code-patch
  suggestion tasks, and document exactly how each fails.

**Environment note for this repo**: no local model runtime (Ollama / llama.cpp / MLX) is
installed on the machine this course is currently being built on. The lab scripts are
written to run correctly once a runtime is available (Module 2 installs one), and will
refuse to fabricate numbers — see [reports/module_01_local_llm_observations.md](../../reports/module_01_local_llm_observations.md).

## Deliverable

`reports/module_01_local_llm_observations.md` — filled in from the three labs above.

## Assessment

You should be able to explain, without notes:

1. Why context length affects memory (KV cache growth), not just "the model gets full."
2. Why Q4 and Q8 quantization of the same model can behave differently in both quality and
   speed.
3. Why a local model can be private but still insecure.
4. Why small models need stricter application architecture (prompting, schemas, retrieval,
   validation) rather than being treated as drop-in replacements for hosted models.

## Further reading

- Model-agnostic — verify current-generation model names against `models/MODEL_CATALOG.md`
  (populated starting Module 3) rather than trusting any fixed list, including this one.
- llama.cpp KV-cache and context documentation (for the runtime-level mechanics referenced
  above).
- Apple's documentation on unified memory architecture, for the hardware grounding in §4.
