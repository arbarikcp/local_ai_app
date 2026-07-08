# Module 8 deliverable — structured output reliability report

Status: **the full production extraction pipeline is built and fully verified against
FakeRuntime, including every reliability layer (constrained decoding, repair retry,
deterministic confidence scoring, review queueing, chunked merging). The real 3-model /
3-mode comparison against actual models is pending the resourced 32GB Mac.**

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `packages/local_ai_core/extraction/schemas.py` | 9 | `InvoiceExtraction` (curriculum's own example, verbatim) and `PersonExtraction`; JSON Schema generation and validation |
| `chunking.py` | 25 | Paragraph-boundary chunking, word-boundary-safe hard splitting (**a real bug caught and fixed here**, below), overlap, and conflict-flagging merge of partial extractions |
| `confidence.py` | 15 | Deterministic, model-independent scoring across every risk-factor combination — explicitly proven to ignore a model's own self-reported `confidence` field |
| `review_queue.py` | 13 | FIFO enqueue/resolve, resolution recording, double-resolve rejection |
| `json_parsing.py` | 7 | Markdown-fence stripping and loose JSON parsing (small intentional duplication of Module 3's logic — see "Boundary decision" below) |
| `pipeline.py` | 25 | Full pipeline: constrained-decoding-first with recorded `FeatureNotSupported` fallback, repair retry (success and exhaustion), confidence integration, review-queue enqueueing, chunked extraction, and **a second real bug caught and fixed here** (below) |
| `scripts/module_08/constrained_decoding_runner.py` | 14 | Golden-set filtering, field-accuracy scoring (including correctly-predicted-null handling), 3-mode comparison orchestration |
| `extraction_eval.py` | 8 | Golden-label evaluation summary statistics |
| `notebooks/08_structured_output_and_extraction.ipynb` | — | **Executed end-to-end** — every reliability layer demonstrated live with real (fake-backed) data, including a caught demo bug (below) |

**116 new tests this module** (660 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Two real bugs caught by this module's own tests

### 1. Word-mangling hard split in `chunking.py`

The first implementation's fallback for a paragraph longer than `max_chars` sliced it by raw
character position (`para[i:i+max_chars]`), which could — and in a test, did — split a word
in half across a chunk boundary (`"lambda."` became `"lam"` + `"bda."` in two different
chunks, silently losing the whole word as a unit in either one). Caught by
`test_no_content_is_lost_without_overlap`, which asserts every word from the original text
appears intact in the chunked output. Fixed by adding `_pack_words()`, which splits on
whitespace and only falls back to a raw character split for the pathological case of a
single word itself longer than `max_chars`.

### 2. `review_queue or ReviewQueue()` silently discarding the caller's queue

`ExtractionPipeline.__init__` used `self.review_queue = review_queue or ReviewQueue()` to
default an unset queue. `ReviewQueue` defines `__len__`, so Python's truthiness treats an
**empty** `ReviewQueue()` as falsy — a caller passing in a fresh, empty queue (exactly the
common case) had it silently replaced by a *different* fresh queue, so entries the caller
expected to see enqueued were invisible to them. Caught by two tests
(`test_exhausting_repair_attempts_leaves_result_unparsed_and_flagged` and
`test_conflicting_chunk_values_flag_for_review`) both asserting `len(queue) == 1` and
getting `0`. Fixed with an explicit `is not None` check. Grepped the rest of the repo for
the same `x or Default()` pattern against any other class defining `__len__`/`__bool__` —
none found elsewhere.

### 3. A notebook demo that didn't demonstrate what it claimed (caught before shipping)

While executing the notebook, the "missing fields queue for review" demo cell printed
`Needs review: False, Pending review items: 0` — the demo used one risk factor (missing
fields) which `confidence.py` correctly scores as `"medium"`, and only `"low"` triggers
review by default. The narrative claimed the cell would show a queued item; the code as
written didn't produce one. Fixed by adding a second risk factor (unconstrained decoding) to
the demo so it actually crosses the review threshold — not a pipeline bug, but a caught gap
between what a demo claimed and what it did, worth recording the same way a code bug is.

## Boundary decision: `json_parsing.py` duplicates ~15 lines from Module 3

`packages/local_ai_core/extraction/json_parsing.py`'s `strip_markdown_fence`/`try_parse_json`
intentionally duplicate `scripts/module_03/scorers/json_validity.py`'s logic rather than
importing it. `packages/` must not depend on `scripts/module_NN/` — the reverse (scripts
importing packages) is the established, expected direction throughout this repo. A ~15-line
duplication was judged the correct trade-off over inverting that layering for one shared
helper.

## Real proof: the reliability ladder works (from the executed notebook)

| Scenario | Confidence | Needs review | Notes |
|---|---|---|---|
| Clean extraction, all fields present | high | no | 0 risk factors |
| Repair retry needed (malformed → fixed) | medium | no | 1 risk factor (repair used) |
| All fields missing + unconstrained decoding | low | **yes** | 2 risk factors → correctly queued |
| Long document, chunked (11 chunks) | — | — | merged correctly to the single true record |

## Lab 8 comparison (from the executed notebook, fake-backed with realistic capability differences)

Using a fake configured to mirror real capability differences from Module 5's
`feature_matrix.py` (JSON-schema support like Ollama, no grammar support like Ollama, and
occasional prompt-only unreliability matching Module 1 §11's small-model behavior):

| Mode | Invalid JSON rate | Field accuracy | Constrained decoding used |
|---|---:|---:|---:|
| text | 50% | 100% | 0% |
| json_schema | 0% | 100% | 100% |
| grammar | 50% | 100% | 0% (falls back to text — this fake has no grammar support) |

**Honesty note on "field accuracy: 100%" alongside "invalid JSON rate: 50%" for text/grammar
modes**: these are not contradictory. Field accuracy is computed only over cases that parsed
successfully — a 50% invalid rate means half the cases are excluded from that average
entirely, not scored as 0%. A report reader should read invalid-JSON-rate as the primary
reliability signal and field-accuracy as "correctness, conditional on the model producing
parseable output at all" — the two numbers answer different questions.

## Known limitations, stated plainly

- **Only 2 of Module 3's 6 golden extraction records match `PersonExtraction`'s schema**
  (`name`/`age`/`city`) — the other 4 model different schemas (invoice, contact, medical,
  order). Lab 8 and Lab 7 both filter to these 2 matching records rather than padding the
  sample with schema-mismatched data. 2 real records is a thin sample for a production
  reliability claim; expanding `PersonExtraction`-shaped golden cases (or adding
  `InvoiceExtraction`-shaped ones) is a natural next step once real model runs are possible.
- **`placeholder_gbnf_grammar()` is not a real, schema-complete grammar.** JSON-Schema-to-GBNF
  generation is a nontrivial ecosystem tool (llama.cpp ships one) and reimplementing it was
  out of scope for this module. The placeholder exists only to exercise the pipeline's
  grammar-request code path and `FeatureNotSupported` handling end to end — it would not
  produce a usable constraint on a real llama.cpp server. Flagged clearly in its own
  docstring and in this report, not left to be discovered as a surprise later.
- **The real 3-model / 3-mode comparison, and the real repair-retry / chunking behavior
  against an actual small model, are pending the resourced 32GB Mac** — same standing
  constraint as every module since Module 1.

## Labs pending live execution

```bash
uv run python scripts/module_08/constrained_decoding_runner.py --model qwen2.5:1.5b
uv run python scripts/module_08/extraction_eval.py --model qwen2.5:1.5b
```

Real Ollama has no grammar support (confirmed in Module 5's `feature_matrix.py`), so the
"grammar" mode row is expected to show `0%` constrained-decoding-used even in a real run —
documented in the script's own skip message so this isn't mistaken for a bug when it happens.
