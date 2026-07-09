# Module 19 — Fine-tuning, LoRA, and Adapters on Mac

> Phase: Advanced · Bible reference: [curriculum.md §29](../../curriculum.md#29-module-19--fine-tuning-lora-and-adapters-on-mac)

## Goal

Teach when and how to customize models locally.

## Decision framework (made real, not just a bulleted list)

```text
Use prompting when: task is simple; behavior can be specified in instructions;
                     examples are few; latency is acceptable.
Use RAG when:        task needs knowledge from private or changing documents;
                     answer must cite sources; knowledge changes frequently.
Use fine-tuning when: output style/format is repetitive; task is narrow and stable;
                     you have enough labeled data; prompt/RAG are not enough;
                     evaluation proves improvement.

Do not fine-tune just to add factual knowledge that changes often. Use RAG.
```

`finetuning/decision_framework.py`'s `recommend_approach()` turns this into one real, testable
function - the same discipline Module 18's `should_use_vlm()` applied to its own decision
diagram.

> **Machine note:** real LoRA fine-tuning needs base model weights and either a GPU or Apple
> Silicon compute this machine intentionally doesn't run a model on
> ([[project-local-ai-app-curriculum]] constraint). `finetuning/mlx_lora.py` wraps `mlx_lm`'s
> real LoRA training/merge API with the same lazy-import/DI pattern as Module 6's `MLXRuntime`
> (`mlx-lm` is already the commented-out `pyproject.toml` entry from that module - it covers
> both text generation and LoRA fine-tuning, no new dependency needed). Everything else this
> module builds - the decision framework, dataset creation/cleaning/splitting, LoRA parameter
> math, overfitting detection, adapter metadata tracking, and the before/after evaluation
> harness - is real, deterministic Python with no model dependency at all.

## Repo structure note

`packages/local_ai_core/finetuning/` (new) joins `runtimes/`, `multimodal/`, etc. as another
foundational, non-RAG-specific subpackage of `local_ai_core`.

## Core topics

### 1. Prompting vs RAG vs fine-tuning

`decision_framework.py`'s `recommend_approach()` - curriculum's own bullet points as real
boolean inputs (`task_is_simple`, `needs_private_or_changing_knowledge`,
`must_cite_sources`, `output_style_is_repetitive`, `has_enough_labeled_data`,
`evaluation_proves_improvement`, ...), returning a real `Recommendation` with the chosen
approach and the specific reason - not a vibe, a traceable decision.

### 2. Instruction tuning

Theory: fine-tuning a model to follow an (instruction, input, output) triple format, the same
shape `dataset.py`'s `TrainingExample` implements for real (§5).

### 3. LoRA

`lora_math.py`'s `lora_trainable_params()` - the real parameter-count formula
(`rank * (d_in + d_out)` per adapted layer, vs. `d_in * d_out` for full fine-tuning of that
layer) - a genuine, computable reduction, not an assertion that "LoRA is more efficient."

### 4. QLoRA conceptually

Theory only: QLoRA quantizes the frozen base model (Module 4's quantization math already covers
the memory side of this) while training LoRA adapters in higher precision on top - not
reimplemented as code, since Module 4 already owns quantization math and this module owns LoRA
math; QLoRA is genuinely the composition of both, not a third thing to build.

### 5. Dataset creation

`dataset.py`'s `TrainingExample` (`instruction`, `input`, `output`) and
`datasets/finetuning/ticket_classification.jsonl` - a real, committed, hand-labeled dataset (40
examples, 4 categories) classifying a support ticket's subject line, continuing this course's
recurring Nimbus support theme (Modules 13, 15-17).

### 6. Data cleaning

`dataset.py`'s `clean_dataset()` - real deduplication (exact-match on `instruction+input`) and
real length filtering (`min_output_chars`/`max_input_chars`), each returning *why* an example
was dropped, not just a smaller list.

### 7. Train/validation/test split

`dataset.py`'s `split_dataset()` - a real, seeded random split at configurable ratios, plus
`detect_leakage()` - a real check that no example's `(instruction, input)` pair appears in more
than one split. Leakage detection is not theoretical here: §"Real proof" in the deliverable
report runs it against a deliberately leaked dataset and catches it.

### 8. Overfitting

`overfitting.py`'s `detect_overfitting()` - a real algorithm over a real (synthetic, since no
training runs here) loss-curve: tracks the best validation loss seen so far and flags
overfitting once validation loss has increased for `patience` consecutive epochs while training
loss kept decreasing - the standard early-stopping signal, implemented as a real, testable
function rather than "watch the loss curves."

### 9. Evaluation before and after

`evaluation.py`'s `compare_before_after()` - reuses Module 13's `answer_metrics` (
`must_contain_score`) to score a golden set's outputs from a baseline and a "fine-tuned" model
side by side, real aggregate deltas. Since no model is actually fine-tuned here, both sides are
`FakeRuntime`-backed with deliberately different scripted quality - honest about being a harness
demonstration, not a real fine-tuning result.

### 10. Adapter management

`adapter_registry.py`'s `AdapterRegistry` - real SQLite persistence (same pattern as Module
8.5's `SessionStore`, Module 14's `AuditLog`), tracking real metadata (base model, rank, alpha,
target modules, dataset hash, file path, creation time) per adapter, proven across an actual
close/reopen cycle.

### 11. MLX fine-tuning path

`mlx_lora.py`'s `train_lora_adapter()` - lazy-imports `mlx_lm.lora`'s real training entry point
inside a function body, `FakeLoraTrainer`-backed for tests, honest-skip for a real run.
"Enabling this for real" instructions in the module docstring, same convention as every other
heavy-dependency adapter since Module 9.

### 12. Merging adapters

`mlx_lora.py`'s `merge_adapter()` - same lazy-import/DI pattern, wrapping `mlx_lm`'s real
adapter-fusion entry point (`mlx_lm.fuse`).

### 13. Fine-tuning small models

Theory, tied to Module 3's model catalog discipline: fine-tuning is scoped to models small
enough to train locally under this course's RAM constraints (curriculum's own framing); no new
model-selection code, Module 3's benchmarking harness already generalizes.

## Hands-on labs

1. **Create labeled dataset** — `datasets/finetuning/ticket_classification.jsonl`,
   `scripts/module_19/dataset_demo.py`.
2. **Fine-tune small model for classification or extraction** — `mlx_lora.py`, honest-skip real
   training, `FakeLoraTrainer`-backed orchestration proven.
3. **Evaluate before/after** — `scripts/module_19/evaluation_demo.py`.
4. **Compare with prompt-only baseline** — same script, a third comparison row.
5. **Package adapter** — `scripts/module_19/adapter_packaging_demo.py`, real SQLite
   registration.
6. **Document failure cases** — same script + the deliverable report's own honest failure
   analysis.

## Deliverable

```text
datasets/finetuning/
  ticket_classification.jsonl
packages/local_ai_core/finetuning/
  decision_framework.py
  dataset.py
  lora_math.py
  overfitting.py
  adapter_registry.py
  mlx_lora.py
  evaluation.py
  tests/
scripts/module_19/
  dataset_demo.py
  evaluation_demo.py
  adapter_packaging_demo.py
reports/module_19_finetuning_report.md
```
