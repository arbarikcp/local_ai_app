# Model Scorecard

Copy this file to `reports/model_scorecard_<model-id>.md` per model benchmarked, filled in
from a real run of `scripts/module_03/run_benchmark.py`. Do not fill in numbers by hand —
every value here should trace back to an actual harness run.

## Model

- Name:
- Version:
- Runtime: (Ollama / llama.cpp / MLX)
- Quantization:
- Context tested:
- RAM tier:
- License notes:

## Performance

| Metric | Value |
|---|---:|
| TTFT p50 | |
| TTFT p95 | |
| tokens/sec | |
| peak memory | |
| invalid JSON rate | |

## Quality

(from `scorers.exact_match`, `scorers.json_validity`, `scorers.rag_metrics` mean scores per task)

| Task | Score | Notes |
|---|---:|---|
| summarization | | |
| extraction | | |
| classification | | |
| code | | |
| rag | | |
| tool_calling | | |

## Recommendation

- Recommended use cases:
- Avoid for:
- Gotchas:
