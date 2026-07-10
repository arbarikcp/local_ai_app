# Project 2 — Production Local RAG Service

> See [PROPOSAL.md](PROPOSAL.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md) for how it's built,
> [REPORT.md](REPORT.md) for measured results, [OUTRO.md](OUTRO.md) for what's next.

## Setup

No install beyond the repo's own `uv sync` (run once at the repo root). This project adds
`psutil` as a real, direct dependency (was already transitive) for the memory eval metric.

## Run the tests

```bash
uv run pytest projects/02_production_rag -q
```

## Run the API service

```bash
uv run uvicorn rag_api:app --app-dir projects/02_production_rag/app --port 8000
```

⚠️ With the default config (`config/app.example.yaml`), this writes to the real
`~/.local-llm-ai/rag/` on your machine (a real LanceDB table and a small SQLite metadata
database) — same disclosed behavior as Project 1 and Module 23. Set
`APP_CONFIG_PATH=/path/to/config.yaml` to point at a different data directory.

The default embedder is `FakeEmbedder` (real bag-of-words hashing, not a neural model) and the
default runtime is `FakeRuntime` (no model runtime installed on this dev machine) — every
`/query` call will come back with `FakeRuntime`'s canned response and no citations until a real
embedder/runtime is injected (see ARCHITECTURE.md's "Machine note" analog — swap via
`build_rag_context(..., embedder=..., runtime=...)`, no other code change needed).

### Example requests

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source_type": "text", "text": "Password reset links expire after 24 hours.", "doc_id": "doc-1"}'

curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source_type": "markdown", "source_path": "datasets/rag_docs/nimbus_handbook/password_reset.md"}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do password reset links last?"}'

curl http://127.0.0.1:8000/documents/doc-1

curl -X DELETE http://127.0.0.1:8000/documents/doc-1

curl -X POST http://127.0.0.1:8000/eval/rag -H "Content-Type: application/json" -d '{}'
```

Real response from the `/query` command above (against the default `FakeEmbedder`/`FakeRuntime`):

```json
{
  "answer": "(FakeRuntime - no model runtime installed on this machine)",
  "citations": [],
  "trace": {"retrieved_chunks": 1, "reranked_chunks": 1, "context_tokens": 9, "model": "llama3.2:3b"}
}
```

## Run the evaluation

```bash
uv run python projects/02_production_rag/evals/run_rag_eval.py
```

Ingests the real, committed 20-document Nimbus handbook corpus
(`datasets/rag_docs/nimbus_handbook/`), then answers every question in the real 10-case golden
set (`evals/rag_golden_set.jsonl`, 8 answerable + 2 deliberately unanswerable) against a
scripted runtime that knows the ground truth — proving the metrics harness works correctly, not
claiming real model quality. See REPORT.md for the actual numbers from a real run.

`POST /eval/rag` runs the same evaluation against whatever is *actually deployed* (the real
ingested corpus and the real configured runtime) rather than a fresh isolated one — an operator
must ingest the relevant corpus first, the same way a real evaluation only means something
against documents actually in production.

## Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health`, `/ready` | GET | Liveness/readiness (Module 23, reused) |
| `/documents` | POST | Ingest a markdown file, PDF, or plain text document |
| `/documents/{doc_id}` | GET | Retrieve stored document metadata and ingestion status |
| `/documents/{doc_id}` | DELETE | Remove a document and its chunks |
| `/query` | POST | Ask a question, get an answer with verified citations and a trace |
| `/eval/rag` | POST | Run the evaluation harness against the real deployed corpus |
