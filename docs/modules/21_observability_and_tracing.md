# Module 21 — Observability and Tracing

> Phase: Production · Bible reference: [curriculum.md §31](../../curriculum.md#31-module-21--observability-and-tracing)

## Goal

Make local AI apps debuggable.

## This module is genuinely new, not a rebuild

Unlike Modules 19-20, `packages/local_ai_core/tracing/` and `packages/local_ai_core/security/`
were scaffolded empty back in Phase 0 and never filled in — this module is the first to build
real code there. Some infrastructure is still reused deliberately: Module 6's `ensure_trace_id()`
(request IDs), `Timer` (real elapsed-time measurement), and the `MetricsHook` Protocol shape
(Module 6, extended by Module 20's `InMemoryMetricsHook`) are the foundation this module's
`MetricsRegistry` and trace spans build on, cited rather than reimplemented.

> **Machine note:** every piece of this module is real, deterministic, testable Python with no
> model dependency — logs, PII redaction, metrics aggregation, trace-span trees, and the
> eval/feedback store all run for real on this machine, no honest-skip surface at all.

## Core topics

### 1. Logs

`structured_logging.py`'s `StructuredLogger` — real structured JSON log emission via stdlib
`logging`, one JSON object per call with a stable field set (`trace_id`, `event`, `fields`).

### 2. Metrics

`metrics_registry.py`'s `MetricsRegistry` — implements curriculum's exact metric table (below)
as real counters and histograms, in-memory, with a real `summary()` aggregation (p50/p95 for
histograms, same percentile math as Module 20's `dashboard.py`).

### 3. Traces

`trace.py`'s `TraceSpan`/`Trace` — a real span tree matching curriculum's trace model exactly
(below), built with a context-manager-style `start_span()` so every span's `elapsed_ms` is real
measured time (Module 6's `Timer`, reused).

### 4. Prompt logging policy

`structured_logging.py`'s `PromptLoggingPolicy` — an explicit, named decision
(`FULL`/`REDACTED`/`HASH_ONLY`/`NONE`) for how much of a prompt a log line may contain, the same
"never a bare default, always a documented decision" discipline Module 6.5's `AdmissionPolicy`
established for concurrency.

### 5. PII redaction

`pii_redaction.py`'s `redact_pii()` — real regex-based detection and redaction of emails, phone
numbers, credit-card-shaped numbers, and SSN-shaped numbers, returning both the redacted text
and a real per-category count (never silently drops evidence that redaction happened).

### 6-7. Token counts, latency metrics

`metrics_registry.py`'s `prompt_tokens`/`completion_tokens`/`request_latency_ms`/`ttft_ms`
histograms — real aggregation over values a caller supplies (this module doesn't count tokens
itself; Module 1's `HFTokenizerCounter` and Module 6's adapters already own that).

### 8-10. Retrieval traces, tool traces, agent step traces

`trace.py`'s `record_retrieval_step()`, `record_tool_call_step()`, `record_agent_step()` —
typed convenience builders over the same generic `TraceSpan`, so a RAG retrieval, a tool call,
and an agent step all end up in one unified trace tree instead of three incompatible logging
shapes.

### 11-12. Evaluation logs, user feedback

`eval_feedback_store.py`'s `EvalFeedbackStore` — real SQLite persistence (same pattern as
Module 8.5's `SessionStore`, Module 19's `AdapterRegistry`) for two append-only tables:
evaluation runs (referencing Module 13's `answer_metrics`/`LocalJudge` scores) and user feedback
(thumbs up/down + optional comment, tied to a `trace_id`).

## Trace model

```text
request_id
  -> input validation
  -> prompt template version
  -> retrieval query
  -> retrieved chunk IDs
  -> reranker scores
  -> context packing
  -> model call
  -> output validation
  -> tool calls if any
  -> final response
  -> evaluation hooks
```

`trace.py`'s `build_request_trace()` produces exactly this shape as a real `Trace` object — one
root span per step, in this order, each with real elapsed time once populated by a lab script
driving a (fake) request end to end.

## Metrics

| Metric | Type | Registry method |
|---|---|---|
| request_count | counter | `increment("request_count")` |
| request_latency_ms | histogram | `observe("request_latency_ms", value)` |
| ttft_ms | histogram | `observe("ttft_ms", value)` |
| tokens_per_second | gauge/histogram | `observe("tokens_per_second", value)` |
| prompt_tokens | histogram | `observe("prompt_tokens", value)` |
| completion_tokens | histogram | `observe("completion_tokens", value)` |
| invalid_json_count | counter | `increment("invalid_json_count")` |
| retrieval_recall_estimate | gauge | `observe("retrieval_recall_estimate", value)` |
| tool_call_count | counter | `increment("tool_call_count")` |
| tool_error_count | counter | `increment("tool_error_count")` |
| policy_violation_count | counter | `increment("policy_violation_count")` |
| fallback_count | counter | `increment("fallback_count")` |

Every metric name in curriculum's table is a real, tested string constant in
`metrics_registry.py`, not a name a caller has to remember to spell correctly.

## Hands-on labs

1. **Add structured logs** — `structured_logging.py`, `scripts/module_21/structured_logs_demo.py`.
2. **Add request IDs** — reuses Module 6's `ensure_trace_id()` unchanged, same script.
3. **Add OpenTelemetry-shaped spans** — `trace.py`, `scripts/module_21/trace_spans_demo.py`. Real
   span tree, real elapsed time; not the real OpenTelemetry SDK (no new heavy dependency for a
   shape this module can already build correctly with stdlib).
4. **Trace RAG retrieval** — `record_retrieval_step()`,
   `scripts/module_21/rag_retrieval_trace_demo.py`.
5. **Trace tool calls** — `record_tool_call_step()`, `scripts/module_21/tool_call_trace_demo.py`.
6. **Build local dashboard or report** — `scripts/module_21/observability_dashboard_demo.py`,
   combining `MetricsRegistry.summary()` and a full request trace into one printed report.

## Deliverable

```text
packages/local_ai_core/tracing/
  structured_logging.py
  pii_redaction.py
  metrics_registry.py
  trace.py
  eval_feedback_store.py
  tests/
scripts/module_21/
  structured_logs_demo.py
  trace_spans_demo.py
  rag_retrieval_trace_demo.py
  tool_call_trace_demo.py
  observability_dashboard_demo.py
reports/module_21_observability_report.md
```
