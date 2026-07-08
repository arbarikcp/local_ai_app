# Module 7 — Prompt Engineering for Small Local Models

> Phase: Application primitives · Bible reference: [curriculum.md §17](../../curriculum.md#17-module-7--prompt-engineering-for-small-local-models)

## Goal

Teach prompt design under weak-reasoning, limited context, and schema reliability
constraints — and build the reusable prompt infrastructure (templates, versioning, few-shot
examples, injection resistance) every later module's prompts run through.

> **Machine note:** this repo is built on a Mac that must never run a model runtime
> ([[project-local-ai-app-curriculum]] constraint; target execution hardware confirmed as a
> separate 32GB Mac). Like Modules 6 and 6.5, the prompt infrastructure itself
> (`packages/local_ai_core/prompts/`) is fully built and tested with no live runtime needed.
> The 3-model comparison (Lab 2) is honest-skip, pending the resourced Mac.

## 1. Why small models need stricter prompts

Revisit Module 1 §11: small (1B-4B) models are more prone to confident fabrication,
instruction drift over long/complex prompts, format collapse under structured-output tasks,
and weak multi-step reasoning. None of this is fixed by better prose in the prompt — it's
fixed by an application architecture that assumes the model will occasionally ignore an
instruction, and validates rather than trusts (Module 6's Gotcha, restated: "do not trust the
output just because the prompt says so").

## 2. System message discipline

The system message carries the model's role and standing rules — it should be short, stable
across requests (feeding Module 6.5's prompt-prefix-reuse and cache-key stability), and never
contain per-request variable content. Mixing task-specific data into the system message
defeats both caching and the model's own learned distinction between "instructions" and
"content."

## 3. Task framing

State the task once, directly, in imperative language. Avoid open-ended framing ("you might
want to...") that a strong model would interpret charitably but a small model may take
literally or ignore.

## 4. Few-shot examples

A small number of well-chosen input→output examples, in the exact output format expected,
teaches format compliance far more reliably than a prose description of the format alone.
Keep examples short (Module 7's design principles below) — long examples eat into the
context budget (Module 1 §6) without proportionally improving compliance.

## 5. Negative examples

Showing a *wrong* output alongside why it's wrong ("do not do this: ... because ...") is a
distinct, complementary tool to positive few-shot examples — useful specifically for common,
predictable failure modes (e.g., wrapping JSON in markdown fences, inventing a missing
field) rather than general task teaching.

## 6. Prompt compression

When the invariant part of a prompt (system + rules + examples) is large relative to the
context budget, shortening it is a direct memory and latency lever (Module 4 §6). Lab 6
measures the actual quality cost of compression rather than assuming it's free or assuming
it's harmless.

## 7. Output constraints

Explicit, enumerable constraints ("Return only valid JSON," "Use null for missing fields,"
"Do not include markdown") reduce the space of acceptable outputs, which is exactly what a
small model needs — vague constraints ("be concise," "format nicely") leave room for
interpretation a small model is more likely to get wrong than a strong one.

## 8. JSON-only prompts

Prompt-level JSON-only instructions are the *weakest* layer of structured-output reliability
— Module 8 will establish constrained decoding as the primary layer, with prompt discipline
as a fallback for runtimes that don't support it. This module builds the prompt half; Module
8 builds the decoding-constraint half.

## 9. Prompt injection resistance

A prompt that concatenates trusted instructions with untrusted content (user input, retrieved
documents, tool output) without a clear boundary invites the model to treat injected
instructions in that content as if they came from the system. `injection_guard.py`'s
`wrap_untrusted_input()` gives untrusted content an explicit, consistent delimiter and a
standing instruction to treat it as data, not commands. Its `scan_for_injection_patterns()`
is a **best-effort heuristic** flagging common attack phrasing — explicitly documented as not
a security boundary by itself; full adversarial treatment is Module 22's job.

## 10. Prompt versioning

Every named prompt has a version, tracked in `PromptRegistry`. This is not bookkeeping for
its own sake — Module 6.5's `response_cache_key()` takes `prompt_version` as a required
parameter specifically so a prompt change invalidates cached responses instead of serving
stale output under a changed prompt (Module 6.5 §11's cache-invalidation rule, now with a
concrete source for the version string).

## 11. Prompt regression tests

Frozen test cases with property-based assertions (not exact-string assertions — Module 6's
Gotcha again) that run against every prompt version, catching the case where a prompt edit
that looks like an improvement silently breaks a previously-passing case.

## Prompt design principles

- Use direct instructions.
- Avoid vague wording.
- Avoid large multi-task prompts.
- Separate reasoning from final output only when necessary.
- Prefer structured schemas.
- Keep examples short.
- Explicitly define unknown behavior (what should the model do if a field is missing? if the
  task doesn't apply? — state it, don't leave it implicit).
- Use deterministic validators.
- Do not trust the output just because the prompt says so.

## Prompt template structure

```text
Role
Task
Input contract
Output contract
Rules
Examples
User input
```

`PromptTemplate.render()` assembles sections in exactly this order — invariant sections
(Role/Task/Input contract/Output contract/Rules/Examples) first, variable content
(User input) last, which is also Module 6.5's prompt-prefix-reuse layout rule, so this
template structure and that caching rule reinforce each other rather than being two
unrelated conventions to remember.

## Example extraction prompt

```text
You are an information extraction engine.

Task:
Extract the requested fields from the input text.

Rules:
- Return only valid JSON.
- Do not include markdown.
- If a field is missing, use null.
- Do not infer values that are not present.
- Follow the schema exactly.

Schema:
{schema}

Input:
{text}
```

## Hands-on labs

1. **Write 5 prompts for the same task** — `scripts/module_07/prompt_variants.py`, an
   extraction task at 5 discipline levels (vague one-liner through the full
   Role/Task/Rules/Schema structure above).
2. **Compare outputs across 3 models** — `scripts/module_07/prompt_runner.py`; honest-skip
   against real models, fully runnable against `FakeRuntime` for infrastructure proof.
3. **Track invalid output rate** — same script, reusing Module 3's `json_validity` scorer.
4. **Add few-shot examples** — `packages/local_ai_core/prompts/few_shot.py`.
5. **Add regression tests** — `scripts/module_07/prompt_eval.py` against frozen cases in
   `evals/prompt_regression/`.
6. **Compress a long prompt and compare quality** — same script's compression-comparison mode.

## Deliverable

```text
packages/local_ai_core/prompts/
  template.py
  registry.py
  few_shot.py
  injection_guard.py
  tests/
scripts/module_07/
  prompt_variants.py
  prompt_runner.py
  prompt_eval.py
evals/prompt_regression/
  extraction_cases.jsonl
reports/module_07_prompt_comparison.md
```

Curriculum's literal deliverable path is a standalone `prompt-lab/` directory; this build
places the reusable template/registry/few-shot/injection infrastructure in
`packages/local_ai_core/prompts/` (Module 6 already scaffolded this subpackage) and the
lab-running scripts under `scripts/module_07/`, per the repo-wide convention. Frozen
regression test cases go in the already-scaffolded `evals/prompt_regression/` rather than a
new `test_cases/` directory.
