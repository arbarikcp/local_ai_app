# Module 6.5 — Serving Concurrency, Batching, and Caching

> Phase: Serving/performance foundation · Bible reference: [curriculum.md §16.5](../../curriculum.md#165-module-65--serving-concurrency-batching-and-caching)

## Goal

Understand how local runtimes schedule concurrent work, why one Mac is often a
single-sequence machine, and how caching avoids recomputation — then build the gateway
infrastructure (bounded queue, response/semantic/embedding caches, admission control) that
turns "concurrency is risky" from folklore into a measured, deliberate policy.

> **Machine note:** this repo is built on a Mac that must never run a model runtime
> ([[project-local-ai-app-curriculum]] constraint; target execution hardware confirmed as a
> separate 32GB Mac). Unlike Modules 1-5 but *like* Module 6, most of this module's
> infrastructure is fully testable without a live runtime: `FakeRuntime` (Module 6) already
> supports simulated latency, which is exactly what's needed to prove queueing and caching
> behavior for real. Only the real per-runtime concurrency measurement (Labs 1-3, 8 against
> an actual Ollama server) is honest-skip, pending the resourced Mac.

## 1. Why local serving differs from cloud inference serving

A cloud inference service scales horizontally — more replicas absorb more concurrent
requests, each still served promptly. A local Mac is one machine with one pool of unified
memory (Module 1 §4): there is no "add another replica." Every concurrent request competes
for the same weights-resident memory, the same KV-cache budget, and the same compute. This
is the single fact this whole module is downstream of.

## 2. Single-sequence vs multi-sequence behavior

Some runtimes process one sequence at a time by default; others support genuine
multi-sequence parallelism (multiple requests' forward passes interleaved). Which mode a
runtime is in changes what "concurrency" even means for it — two requests to a
single-sequence runtime aren't parallelized, they're serialized with worse bookkeeping than
just queueing them yourself, which is exactly why this module builds an explicit queue
(§"Deliverable") rather than trusting the runtime to do the right thing implicitly.

## 3. Request queueing vs rejection

Two honest responses to "too much concurrent demand": **queue** the request (bounded, with a
max depth) or **reject** it immediately (fail fast). An unbounded queue is not a third
option — it's rejection with extra steps and worse latency, since a request that waits
forever behind an unbounded backlog is functionally the same as a rejected one, just slower
to fail. `gateway/queue.py`'s `BoundedRequestQueue` implements both: admit up to
`max_concurrent` running at once, queue up to `max_queue_size` waiting, reject beyond that.

## 4. Ollama concurrency knobs

```bash
OLLAMA_NUM_PARALLEL       # concurrent requests served per model
OLLAMA_MAX_LOADED_MODELS  # how many distinct models stay resident
OLLAMA_KV_CACHE_TYPE      # f16 | q8_0 | q4_0, depending on runtime support
keep_alive                # per-request/model residency control (Module 5 §6-7)
```

Loading an embedder and generator simultaneously can double resident model weights — on
8-16GB Macs this is often the actual out-of-memory cause, not context length (Module 4 §"Reranker
and embedder memory contention" already flagged the reranker version of this).

## 5. llama.cpp parallel slots and continuous batching

```bash
--parallel N / -np N      # N slots = N concurrent sequences
--cont-batching           # continuous batching where supported
--ctx-size C              # total context budget
```

## 6. Context-per-slot traps

**Gotcha**: with `-np 4` and `--ctx-size 8192`, each request may effectively get about 2048
tokens of usable context — llama.cpp-style servers often divide the total context budget
across parallel slots. Increasing parallelism can silently shrink every request's context
budget. This is precisely the kind of runtime-specific behavior Module 5's `feature_matrix.py`
exists to make explicit rather than assumed.

## 7-10. Caching: response, semantic, KV-prefix, embedding

| Cache type | Key | Hit saves | Watch out |
|---|---|---|---|
| Response cache | hash(model, rendered prompt, params, prompt version) | entire generation | invalidation on prompt/model/version change |
| Semantic cache | embedding(query) above similarity threshold | generation on near-duplicate queries | false hits; threshold must be tuned and audited |
| KV prefix reuse | stable system + schema + examples prefix | prompt-eval time and TTFT | runtime support varies (Module 5 §8) |
| Embedding cache | hash(text, embedding model, normalization version) | re-embedding during indexing | invalidate on embedding-model change |

**Prompt layout rule**: put the invariant part first —

```text
stable system prompt
stable safety policy
stable output schema
stable few-shot examples
variable user/document content
```

This helps runtimes that can reuse prompt prefixes (KV-prefix reuse, §9) and makes rendered
prompt snapshot tests easier to review. `gateway/cache.py` implements response and semantic
caching directly (Labs 5-6); KV-prefix reuse is a runtime-level behavior this module
documents but doesn't implement (it's not something application code controls directly);
embedding caching (Lab 7) is implemented for the future ingestion pipeline Module 9-11 will
build.

## 11. Cache invalidation

**Cache keys must include model, quantization, prompt version, tool version, schema version,
and safety policy version when those affect output** (Gotcha, verbatim from the bible). A
response cache keyed only on `(model, prompt)` will happily serve a stale answer after a
prompt template change, a schema change, or a safety policy update — `gateway/cache.py`'s
`response_cache_key()` takes all of these as explicit parameters rather than optional
extras, so omitting one is a visible choice, not an accident.

## 12. Thermal throttling and backpressure

Sustained high concurrency on a laptop-class Mac can trigger thermal throttling — the chip
reduces clock speed to manage heat, which shows up as *declining* tokens/sec over a long run,
not just higher latency at the start. Module 4's Lab 4.3 already asked for this to be
observed manually; this module's admission control (§"Why `max_concurrent_requests: 1` is
often correct") is the architectural response: don't let demand reach the point where thermal
throttling becomes the bottleneck in the first place.

## Why `max_concurrent_requests: 1` is often correct

On a single unified-memory Mac, concurrency multiplies KV-cache pressure (Module 4 §7-8) and
can trigger swap, fan noise, thermal throttling, and worse p95 latency. The honest production
default is usually:

```yaml
max_concurrent_requests: 1
```

Then increase to 2 only *after measurement* — `gateway/admission_control.py`'s
`AdmissionPolicy` defaults to exactly this, with a `reason` field that must be updated to
cite the measurement that justified any higher value, so the policy is a decision on record,
not a magic constant nobody can explain later.

## Hands-on labs

1. **Run 1, 2, and 4 concurrent requests against the same model** —
   `scripts/module_06_5/lab_measure_concurrency.py`, extending Module 4's raw concurrency
   simulation with this module's actual queue/admission-control layer. Needs a live runtime;
   honest-skip on this machine.
2. **Compare native runtime concurrency settings** — same script, documents Ollama's
   `OLLAMA_NUM_PARALLEL` vs. llama.cpp's `-np` behavior differences (§4-6 above).
3. **Measure queue wait, TTFT, total latency, tokens/sec, memory, and failure rate** — same
   script; `BoundedRequestQueue` reports queue wait separately from execution time
   specifically so this measurement is possible.
4. **Add a bounded request queue** — `gateway/queue.py`. Fully built and tested here, no
   runtime needed.
5. **Add response cache** — `gateway/cache.py`'s `ResponseCache`. Fully built and tested.
6. **Add semantic cache with a conservative similarity threshold** — `gateway/cache.py`'s
   `SemanticCache`, defaulting to a high (0.95) threshold per the Gotcha below. Fully built
   and tested.
7. **Add embedding cache to the ingestion pipeline** — `gateway/cache.py`'s
   `EmbeddingCache`, ready for Module 9-11's ingestion pipeline to use. Fully built and
   tested.
8. **Show before/after latency on a repeated-query workload** —
   `scripts/module_06_5/lab_caching_before_after.py`, run against `FakeRuntime` with
   simulated latency. This lab needs **no live runtime at all** to produce a real,
   non-fabricated before/after number, unlike Labs 1-3.

## Gotchas

- Concurrency is not free; it often improves average throughput while making p95 latency
  worse — `gateway/admission_control.py`'s `recommend_policy_from_measurements()` checks p95,
  not just mean latency, for exactly this reason.
- Continuous batching can help throughput, but it does not remove memory limits (Module 4's
  memory math still applies at every concurrency level).
- Semantic caching can return confidently wrong answers for near-but-not-equivalent
  questions — `SemanticCache` defaults its similarity threshold conservatively high (0.95)
  and returns the matched similarity score alongside every hit so a caller can audit borderline
  matches rather than trusting the cache blindly.
- Cache keys must include model, quantization, prompt version, tool version, schema version,
  and safety policy version when those affect output (§11 above, `response_cache_key()`'s
  signature enforces this by taking them as required parameters).

## Deliverable

```text
packages/local_ai_core/gateway/
  queue.py
  cache.py
  admission_control.py
  tests/
scripts/module_06_5/
  lab_measure_concurrency.py
  lab_caching_before_after.py
reports/module_06_5_serving_concurrency_report.md
```

The report must include measured latency/queueing at 1/2/4 concurrent requests under at
least two runtime settings, plus a before/after result for response and semantic caching.
The caching before/after result is real (Lab 8, `FakeRuntime`-backed); the 1/2/4-concurrency
runtime measurement is pending the resourced Mac, per this module's machine note.
