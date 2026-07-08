# Model Catalog

Tracks candidate local models for this course, by category and RAM tier. Populated in
Module 3 ([docs/modules/03_local_model_selection_and_benchmarking.md](../docs/modules/03_local_model_selection_and_benchmarking.md)).

## How to read this file

- Every entry uses the schema from curriculum.md §6.4. Fields left `TBD` have not been
  verified against the actual running model — see the "Verification status" note on each
  entry before trusting it.
- **License is not legal advice.** `license_notes` records what to check, not a cleared
  determination. Re-verify before any commercial use.
- **This catalog was populated on a machine that does not run local models**
  ([[project-local-ai-app-curriculum]] constraint) — `runtime`, `quantization_tested`,
  `context_tested`, and `known_issues` reflect public documentation and the course bible's
  own guidance, not measurements taken here. `last_verified` records when the *catalog
  entry's public information* was checked, not when the model was benchmarked locally.
  Module 3's benchmark harness (`scripts/module_03/run_benchmark.py`) is what turns a
  catalog entry into a measured scorecard, once run on a resourced Mac.

## Verification status legend

- 📄 **documented** — filled from public model card / documentation, not run here.
- 📊 **benchmarked** — has a corresponding `reports/model_scorecard_*.md` from an actual run.

---

## General instruction/chat models

```yaml
model_id: qwen2.5:1.5b-instruct
family: qwen2.5
category: chat
runtime:
  ollama: true
  gguf: true
  mlx: true
recommended_ram_tier: 8gb
quantization_tested: []       # TBD — not benchmarked on this machine
context_tested: []            # TBD
use_cases:
  - classification
  - short summarization
  - routing
known_issues:
  - smaller instruct models are more prone to instruction drift on multi-part prompts (Module 1 §11)
license_notes: "verify exact license per Qwen2.5 release; historically Apache-2.0 for many open-weight releases, but confirm per model/variant before commercial use"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: qwen2.5:7b-instruct
family: qwen2.5
category: chat
runtime:
  ollama: true
  gguf: true
  mlx: true
recommended_ram_tier: 16gb
quantization_tested: []
context_tested: []
use_cases:
  - production RAG generation
  - structured extraction
  - local chat
known_issues: []
license_notes: "verify exact license per Qwen2.5 release before commercial use"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: llama3.1:8b-instruct
family: llama-3.1
category: chat
runtime:
  ollama: true
  gguf: true
  mlx: true
recommended_ram_tier: 16gb
quantization_tested: []
context_tested: []
use_cases:
  - general chat
  - RAG generation
known_issues:
  - Meta Community License includes acceptable-use policy and scale-based restrictions; not OSI-open
license_notes: "Meta Community License — read the acceptable-use policy before any commercial or high-scale use"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: gemma2:9b-instruct
family: gemma2
category: chat
runtime:
  ollama: true
  gguf: true
  mlx: maybe
recommended_ram_tier: 16gb
quantization_tested: []
context_tested: []
use_cases:
  - general chat
  - summarization
known_issues:
  - verify current Gemma generation at integration time — do not hard-code this family name into application code (curriculum.md §6.2)
license_notes: "Gemma Terms of Use — use restrictions apply, not OSI-open"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: phi3.5:mini-instruct
family: phi-3.5
category: chat
runtime:
  ollama: true
  gguf: true
  mlx: maybe
recommended_ram_tier: 8gb
quantization_tested: []
context_tested: []
use_cases:
  - lightweight assistants
  - routing
known_issues: []
license_notes: "often permissive for recent small Phi releases — verify per release and artifact source"
last_verified: 2026-07-08
verification_status: documented
```

## Code models

```yaml
model_id: qwen2.5-coder:1.5b
family: qwen2.5-coder
category: code
runtime:
  ollama: true
  gguf: true
  mlx: maybe
recommended_ram_tier: 8gb
quantization_tested: []
context_tested: []
use_cases:
  - code explanation
  - small patch suggestions
known_issues:
  - may hallucinate APIs; needs strict output schema for patch format (curriculum.md §6.4 example)
license_notes: "verify before commercial use"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: qwen2.5-coder:7b
family: qwen2.5-coder
category: code
runtime:
  ollama: true
  gguf: true
  mlx: maybe
recommended_ram_tier: 16gb
quantization_tested:
  - q4_k_m
  - q5_k_m
context_tested:
  - 4096
  - 8192
use_cases:
  - code explanation
  - test generation
  - patch proposal
known_issues:
  - may hallucinate APIs
  - needs strict output schema for patch format
license_notes: "verify before commercial use"
last_verified: 2026-07-08
verification_status: documented
```

## Embedding models

```yaml
model_id: nomic-embed-text
family: nomic-embed-text
category: embedding
runtime:
  ollama: true
  gguf: false
  mlx: false
  sentence_transformers: true
recommended_ram_tier: 8gb
quantization_tested: []
context_tested: []
use_cases:
  - RAG retrieval
  - semantic search
known_issues:
  - benchmark Ollama's embedding endpoint against sentence-transformers directly before committing — throughput/quality can differ (curriculum.md §19)
license_notes: "verify current license before commercial use"
last_verified: 2026-07-08
verification_status: documented
```

```yaml
model_id: BAAI/bge-small-en-v1.5
family: bge
category: embedding
runtime:
  ollama: false
  gguf: false
  mlx: false
  sentence_transformers: true
recommended_ram_tier: 8gb
quantization_tested: []
context_tested: []
use_cases:
  - RAG retrieval
  - small-footprint embedding
known_issues: []
license_notes: "MIT-style license historically for BGE-small — verify per release"
last_verified: 2026-07-08
verification_status: documented
```

## Reranking models

```yaml
model_id: BAAI/bge-reranker-base
family: bge-reranker
category: reranker
runtime:
  ollama: false
  gguf: false
  mlx: false
  sentence_transformers: true
recommended_ram_tier: 16gb
quantization_tested: []
context_tested: []
use_cases:
  - RAG reranking
known_issues:
  - cross-encoder rerankers add a third resident model alongside generator + embedder during a RAG query — watch memory contention on 8-16GB tiers (curriculum.md §14, reranker RAM contention)
license_notes: "verify per release"
last_verified: 2026-07-08
verification_status: documented
```

---

## Coverage gaps (not yet catalogued)

- Multimodal/vision models (Module 18 will populate these when that module starts).
- Any model actually pulled/run and benchmarked — every entry above is 📄 documented, not
  📊 benchmarked. The first 📊 entries will come from running
  `scripts/module_03/run_benchmark.py` against real models on a resourced Mac.
