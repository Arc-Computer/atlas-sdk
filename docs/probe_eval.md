## Capability Probe LLM Evaluation Plan

Goal: compare several LLM backends (Gemini 2.5 Flash, Claude Haiku 4.5, Grok 4 Fast) for routing accuracy, latency, and cost within the capability probe.

### Inputs
- JSONL dataset where each row contains:
  - `task`: string prompt sent to the probe.
  - `learning_history`: aggregated payload (matches `ExecutionContext.metadata["learning_history"]`).
  - `expected_mode` *(optional)*: ground-truth adaptive mode chosen during the recorded run.
  - `metadata` *(optional)*: arbitrary dictionary for tagging/grouping.
- API keys for Google, Anthropic, and xAI available via `.env`.

The repository ships a starter dataset at `atlas/data/sample_probe_payloads.jsonl` (3 anonymised examples) that can be extended with real traces.

### Metrics
For each model:
- Decision distribution (count per mode).
- Accuracy vs. `expected_mode` (when provided).
- Average and percentile latency (wall-clock).
- Failure rate (exceptions/timeouts).
  

### Procedure
1. Load the dataset and normalise each record into probe inputs.
2. For each candidate model, instantiate a `CapabilityProbeClient` with custom `LLMParameters`.
3. Sequentially (or with limited concurrency) evaluate all samples, timing each request.
4. Aggregate metrics and output a comparative table plus JSON report for further analysis.
5. Optionally, export raw decisions for manual review.

### Running the evaluation
```
python -m scripts.eval_probe_models \
  --dataset atlas/data/sample_probe_payloads.jsonl \
  --models gemini anthropic xai
```

Environment variables can override specific model identifiers:
```
export ATLAS_PROBE_MODEL_GEMINI=gemini/gemini-2.5-flash
export ATLAS_PROBE_MODEL_ANTHROPIC=anthropic/claude-haiku-4-5
export ATLAS_PROBE_MODEL_XAI=xai/grok-4-fast
```

The script automatically loads `.env`, so ensure `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and `XAI_API_KEY` are populated before running it.

### Capturing a dataset

Use `scripts/export_probe_dataset.py` to dump probe inputs from your Postgres instance. The `--per-learning-key-limit` flag keeps only one sample per learning key so repeated sessions do not skew the evaluation:

```bash
python -m scripts.export_probe_dataset \
  --database-url "$STORAGE__DATABASE_URL" \
  --output data/probe_eval.jsonl \
  --limit 1000 \
  --min-history 3 \
  --per-learning-key-limit 1
```

Append multiple exports (varying `--offset`) if you need more coverage, then feed the merged JSONL file into the evaluation command above.

### Deliverables
- CLI script producing JSON/markdown summaries.
- Optional pytest to ensure the harness handles datasets without hitting external APIs (uses stubs).
- Documentation describing usage and extension guidelines (this file).
