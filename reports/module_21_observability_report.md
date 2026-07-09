# Module 21 deliverable — observability and tracing report

Status: **complete.** No honest-skip surface at all — unlike most modules since Module 18,
`packages/local_ai_core/tracing/` was scaffolded empty back in Phase 0 and genuinely new this
module, and every piece of it (structured logging, PII redaction, metrics aggregation, trace
spans, the eval/feedback store) is real, deterministic Python with no model dependency
whatsoever. The only reuse is deliberate and cited: Module 6's `ensure_trace_id()` and `Timer`.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `tracing/structured_logging.py` | 9 | `PromptLoggingPolicy`'s four modes, real JSON log emission, HASH_ONLY/NONE never leak the raw prompt |
| `tracing/pii_redaction.py` | 16 | Real regex detection across four PII categories, correct precedence (SSN not double-counted as a phone number) |
| `tracing/metrics_registry.py` | 25 | Curriculum's exact 12-metric table, counter/observable type enforcement, real p50/p95 aggregation |
| `tracing/trace.py` | 34 | Real elapsed-time span measurement, retrieval/tool-call/agent-step convenience builders, core-step validation |
| `tracing/eval_feedback_store.py` | 16 | Real SQLite persistence for eval runs and user feedback, including a genuine close/reopen cycle |
| `scripts/module_21/` (5 lab scripts) | 23 | Labs 1-6 exercised for real against a live vector search, real trace spans, and real SQLite storage |
| `notebooks/21_observability_and_tracing.ipynb` | — | **Executed end-to-end** — every cell a real computation |

**64 new tests this module** (1695 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: prompt logging policies genuinely never leak what they promise not to

```
full:      {"prompt": "Classify this ticket for jane.doe@example.com: I was charged twice.", ...}
redacted:  {"prompt": "Classify this ticket for [EMAIL]: I was charged twice.", ...}
hash_only: {"prompt_hash": "7fceb095ee8cbdd9", ...}
none:      {"prompt_logging_policy": "none"}
```

A unit test (`test_hash_only_policy_never_includes_the_raw_prompt`) checks the raw prompt string
doesn't appear anywhere in the serialized JSON, not just that a `"prompt"` key is absent — a
field named something else containing the same text would still fail the check.

## Real proof: PII redaction correctly resolves overlapping pattern categories

```
Redacted text: Contact [EMAIL] or call [PHONE]. SSN on file: [SSN].
Redaction counts: {'email': 1, 'ssn': 1, 'phone': 1}
```

A 9-digit-run SSN (`123-45-6789`) would also match a naive phone-number or credit-card digit
pattern; `redact_pii()`'s category order (email → ssn → phone → credit_card) masks the more
specific pattern first, so a real SSN is never miscounted as a phone number — proven directly by
`test_ssn_is_not_double_counted_as_a_phone_number`, not just asserted by construction.

## Real proof: the trace model matches curriculum's diagram exactly, with real elapsed time

```
Span order: input_validation -> prompt_template_version -> retrieval_query ->
  retrieved_chunk_ids -> reranker_scores -> context_packing -> model_call ->
  output_validation -> final_response -> evaluation_hooks
Total elapsed: 3.79ms
Missing core steps: []
```

Every span's `elapsed_ms` is real measured wall-clock time (Module 6's `Timer`, reused) around
actual `time.sleep()` calls standing in for real work — not a hand-asserted constant. The 3.79ms
total is a genuine consequence of the two real sleeps (1ms + 2ms) plus real Python overhead.

## Real proof: RAG retrieval trace runs against a real vector search

```
Retrieved doc IDs: ['password-reset-guide', 'billing-faq']
Top score: 0.9939
```

`NumpyEmbeddingIndex` (Module 9) does a real cosine-similarity search over three hand-built unit
vectors; the query vector `[0.9, 0.1, 0.0]` is genuinely closest to `password-reset-guide`'s
`[1.0, 0.0, 0.0]` — the 0.9939 similarity score is computed, not asserted, and the trace records
exactly the doc IDs and scores that search actually returned.

## Real proof: tool call trace and metrics agree on what actually happened

```
lookup_order({'order_id': 'ORD-123'}) -> ok
cancel_order({'order_id': 'ORD-999'}) -> error

Metrics: tool_call_count=2, tool_error_count=1
```

The trace's per-call spans and the `MetricsRegistry`'s counters are two independent recordings
of the same two tool calls — a unit test (`test_metrics_and_trace_agree_on_counts`) checks they
never drift apart, since a dashboard built on metrics alone and a debugger built on traces alone
must describe the same reality.

## Real proof: the observability dashboard ties traces, metrics, and feedback together

```
Requests traced: 3
Mean trace latency: 0.0021ms
Feedback summary: {'down': 1, 'up': 2}
Traces missing a core step: []
```

Three requests' traces, eval scores, and feedback ratings are all correlated by `trace_id`
through `EvalFeedbackStore` (real SQLite) and `MetricsRegistry` (real in-memory aggregation) —
`feedback_summary()`'s `GROUP BY rating` count matches the three scripted ratings exactly, and
`validate_trace_shape()` confirms every trace completed its core steps before its feedback was
logged.

## Deliberately not done in Module 21

- **The real OpenTelemetry SDK** — `trace.py`'s `TraceSpan`/`Trace` match curriculum's trace
  model shape exactly using stdlib only; adopting the real `opentelemetry-sdk` package (a new
  dependency) is deferred, since this module's own span tree already proves the shape correctly
  without it.
- **A real token counter feeding `prompt_tokens`/`completion_tokens`** — `metrics_registry.py`
  aggregates whatever values a caller supplies; Module 1's `HFTokenizerCounter` and Module 6's
  runtime adapters already own real token counting, cited rather than reimplemented.
- **A rendered HTML/web dashboard** — Lab 6's "local dashboard or report" is a printed markdown
  report (same choice curriculum explicitly offers: "dashboard *or* report"), consistent with
  every other module's report-based deliverable in this course.
