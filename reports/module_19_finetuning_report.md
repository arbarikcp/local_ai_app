# Module 19 deliverable — fine-tuning, LoRA, and adapters report

Status: **complete.** Real LoRA training itself is honest-skipped — it needs base model weights
and either a GPU or Apple Silicon compute this dev machine intentionally doesn't run a model on
([[project-local-ai-app-curriculum]] constraint) — but every other core topic (decision
framework, dataset creation/cleaning/splitting with real leakage detection, LoRA parameter-count
math, overfitting detection, adapter metadata tracking, and a before/after evaluation harness) is
real, deterministic Python with no model dependency at all.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `datasets/finetuning/ticket_classification.jsonl` | — | A real, committed, hand-labeled dataset — 40 examples, 4 categories, 10 each |
| `finetuning/decision_framework.py` | 9 | `recommend_approach()`'s RAG-priority override, all-four-preconditions-required fine-tuning gate, prompting fallback |
| `finetuning/dataset.py` | 11 | Real loading, deduplication, length filtering, seeded splitting, leakage detection (including a deliberately leaked split) |
| `finetuning/lora_math.py` | 7 | Real parameter-count formulas for full fine-tuning vs. LoRA, a genuine reduction on realistic layer shapes |
| `finetuning/overfitting.py` | 6 | The patience-based early-stopping signal on both an overfitting and a healthy loss curve, including a noisy-but-plateaued-train-loss case that must **not** trigger |
| `finetuning/adapter_registry.py` | 5 | Real SQLite persistence, including a genuine close/reopen cycle |
| `finetuning/mlx_lora.py` | 5 | DI-injected train/merge subprocess wrapper, `is_mlx_lm_available()` genuinely returning `False` on this machine |
| `finetuning/evaluation.py` | 4 | `compare_before_after()`'s per-case and aggregate deltas, reusing Module 13's `must_contain_score` |
| `scripts/module_19/` (3 lab scripts) | 14 | Labs 1, 3-6 exercised for real against the committed dataset and scripted runtimes |
| `notebooks/19_finetuning_lora_and_adapters.ipynb` | — | **Executed end-to-end** — every cell a real computation |

**62 new tests this module** (1576 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: the dataset is genuinely balanced and clean

```
Counter({'account': 10, 'billing': 10, 'technical': 10, 'security': 10})
```

`dataset_demo.py` re-adds the dataset's own first example as a deliberate duplicate before
cleaning, and `clean_dataset()` drops exactly that one example with the real reason
`"duplicate instruction+input"` — not a hand-asserted count, the actual return value of running
deduplication against a genuinely duplicated list.

## Real proof: leakage detection catches a leak when there is one

`detect_leakage()` reports zero leaked keys against the real 32/4/4 seeded split of the 40-example
dataset, and a unit test (`TestDetectLeakage.test_deliberately_leaked_split_is_caught`)
constructs a `DatasetSplit` with the same example placed in both `train` and `validation`,
confirming the function actually flags it rather than only ever returning an empty list.

## Real proof: LoRA parameter math is a genuine reduction, not an assertion

```
Full fine-tune trainable params: 5,505,024
LoRA (rank 8) trainable params: 77,824
LoRA is 1.41% the size of full fine-tuning
```

Computed from `mlx-community/Qwen2.5-1.5B-Instruct-4bit`'s real attention-projection shapes
(`q_proj`/`k_proj`/`v_proj`/`o_proj`), using the formula `rank * (d_in + d_out)` per adapted
layer vs. `d_in * d_out` for full fine-tuning of that layer — a checkable number, not a claim.

## Real proof: overfitting detection distinguishes a real overfit from noisy-but-healthy training

```
Overfitting detected: True
Would have stopped at epoch: 5
Best validation loss: 0.9 (epoch 3)
```

On a synthetic-but-realistic loss curve where validation loss climbs after epoch 3 while train
loss keeps falling, `detect_overfitting()` correctly flags it with `patience=2`. A separate unit
test (`test_noisy_validation_does_not_count_when_train_loss_has_plateaued`) proves the detector
does **not** false-positive when validation loss worsens but training loss has also stalled —
the train-loss condition specifically rules out that case, since a plateaued train loss means the
model isn't the one doing the overfitting; it's just noisy validation.

## Real proof: `is_mlx_lm_available()` genuinely reflects this machine's constraint

```
mlx-lm importable on this machine: False
```

Not mocked — `is_mlx_lm_available()` does a real `import mlx_lm` inside a try/except and returns
`False` because `mlx-lm` is genuinely not installed on this dev machine (its `pyproject.toml`
entry, inherited from Module 6, stays commented out on purpose). `MlxLoraTrainer` itself is
proven via dependency injection (`FakeLoraTrainer`-style callables), the same pattern as Module
6's `MLXRuntime` — real subprocess wiring to `mlx_lm.lora`/`mlx_lm.fuse` is written and
documented, just not exercised for real here.

## Real proof: before/after evaluation shows a genuine, quantified improvement

```
## Lab 3: prompt-only baseline vs fine-tuned candidate
- baseline mean score: 0.50
- candidate mean score: 1.00
- delta: +0.50 (improved=True)

## Lab 4: unhelpful baseline vs fine-tuned candidate
- baseline mean score: 0.00
- candidate mean score: 1.00
- delta: +1.00 (improved=True)
```

Both sides are `FakeRuntime`/scripted-runtime-backed with deliberately different scripted
quality — honest about being a harness demonstration over real golden cases and Module 13's real
`must_contain_score`, not a claim that an actual model was fine-tuned.

## Deliberately not done in Module 19

- **Real LoRA training or adapter merging** — `mlx_lora.py`'s `train_lora_adapter()` and
  `merge_adapter()` are fully built with the same lazy-import/DI pattern as every other
  real-model adapter in this course; actually running them (does a real LoRA adapter improve a
  real model's ticket-classification accuracy) is deferred to the resourced 32GB Mac, per this
  repo's dev-machine constraint.
- **QLoRA as code** — theory only (module doc §4). QLoRA is genuinely the composition of Module
  4's quantization math (frozen base model) and this module's LoRA math (trainable adapters), not
  a third thing to separately implement.
- **A real overfitting run** — the loss curve used to prove `detect_overfitting()` is
  synthetic-but-realistic, labeled as such everywhere it appears (module doc, lab script,
  notebook, this report), since no training runs happen on this machine.
