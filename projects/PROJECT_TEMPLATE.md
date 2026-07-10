# Project structure convention

Every project under `projects/<NN_name>/` (Projects 1-5 and the capstone) follows this same
structure, so a reader who's read one project's docs already knows where to find anything in
the next one.

```text
projects/<NN_name>/
  PROPOSAL.md       — why this project, how we'll build it, what it achieves, how success is measured
  ARCHITECTURE.md   — high-level (diagram, data flow) + low-level (component contracts, storage, API)
  app/              — the real service code
  schemas/          — Pydantic schemas
  prompts/          — prompt templates
  evals/            — evaluation harness + labeled dataset
  tests/            — unit/integration tests
  README.md         — how to run (setup, commands, examples)
  REPORT.md         — measured results against PROPOSAL's success metrics
  OUTRO.md          — what was achieved, what's next, what new tech to explore
```

## What goes in each document

- **PROPOSAL.md** — written first, before code. States the problem (why this project exists),
  the approach (how it will be built, what it reuses from earlier modules vs. builds fresh), the
  objective (what "done" looks like), and the success metrics (how it will be measured) —
  curriculum's own "Evaluation" section for that project, restated as a commitment made before
  building rather than a summary written after.
- **ARCHITECTURE.md** — a high-level section (the data-flow diagram, the deployment shape,
  which earlier modules' components are reused where) and a low-level section (exact function
  signatures, storage schema, API contract, error handling) — a reader should be able to predict
  the code from this document, not just get a vibe for the system.
- **app/**, **schemas/**, **prompts/**, **evals/**, **tests/** — curriculum's own deliverable
  layout (curriculum.md's per-project "Deliverables" section), used verbatim so a reader
  familiar with the bible finds things where they expect.
- **README.md** — practical, imperative: setup, the exact commands to run the service and the
  evaluation, example requests/responses. No narrative, no "why" — that's PROPOSAL.md's job.
- **REPORT.md** — filled in after building, real measured numbers against every metric
  PROPOSAL.md committed to, plus real bugs found and fixed and an honest account of what's
  honest-skip on this dev machine (no LLM runtime installed) vs. what's deferred to the
  resourced 32GB Mac.
- **OUTRO.md** — a short retrospective: what this project achieved, what its own honest-skip
  surface still leaves undone, and what new technique/technology would be the natural next
  thing to explore from here (not a todo list — a pointer for a reader deciding what to learn
  next).

## Naming convention inside `app/`, `schemas/`, `prompts/`, `evals/`, `tests/`

Every project's `app/`, `schemas/`, `prompts/`, `evals/`, and `tests/` directories get their own
entry in `pyproject.toml`'s pytest `pythonpath` list (same mechanism as `scripts/module_NN/`),
so their modules are importable by filename alone (e.g. `import storage`). Because pytest
collects and imports every project's tests in **one shared process**, two projects both naming a
file `storage.py` or a test `test_storage.py` would collide — Python's module cache and pytest's
duplicate-basename check are both process-global, not scoped per project.

**Rule: every importable filename inside a project must be prefixed with that project's own
domain word** (e.g. Project 1's extraction service uses `extraction_storage.py`,
`extraction_service.py`, `test_extraction_storage.py`; Project 2's RAG service would use
`rag_storage.py`, `rag_service.py`; and so on). The outer directory names (`app/`, `schemas/`,
`prompts/`, `evals/`, `tests/`) stay identical across projects — only the files inside need a
project-specific prefix. This keeps the structure genuinely consistent (same shape, discoverable
the same way) without the hidden collision risk generic names would create.

## Why this order

PROPOSAL and ARCHITECTURE are written *before* code, the same discipline this course's modules
already apply informally (a theory doc before the lab scripts) - made explicit and mandatory for
project-scale work, where the temptation to start coding before deciding what "done" means is
much stronger than in a single-module lab.
