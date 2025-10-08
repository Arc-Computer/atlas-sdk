# ARC-AGI Demo Walkthrough

This guide shows how to run the ARC Bring-Your-Own-Agent demo using the `arc-atlas` PyPI package. It assumes a Unix-like shell and Python 3.10+.

## 1. Environment Setup

```bash
python -m venv demo-env
source demo-env/bin/activate
pip install --upgrade pip
pip install arc-atlas openai httpx python-dotenv
```

Export your OpenAI credentials (or configure `OPENAI_BASE_URL` if pointing at a compatible endpoint):

```bash
export OPENAI_API_KEY="sk-..."
# Optional overrides
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export ATLAS_ARC_DEMO_MODEL="o4-mini"
# export ATLAS_ARC_DEMO_REASONING="medium"
```

The repository includes a sample ARC training task at `examples/arc_agi_demo/data/arc_training_0ca9ddb6.json`. You can swap in any other task by passing `--task-id`.

## 2. Inspect a Baseline Run

Run the agent directly to see its unassisted output:

```bash
python examples/arc_agi_demo/scripts/run_baseline.py \
  --task-file examples/arc_agi_demo/data/arc_training_0ca9ddb6.json
```

Use `--task-id <training-id>` to pull any other ARC puzzle from the public repo.

## 3. Run the Atlas Orchestration

```bash
python examples/arc_agi_demo/scripts/run_demo.py \
  --config examples/arc_agi_demo/configs/arc_agi_demo.yaml \
  --task-id 0ca9ddb6 --stream-progress
```

The script downloads the requested ARC training task, converts it into a text prompt, and runs the Atlas Teacher → Student → Reward loop. The console displays the plan, intermediate telemetry, and final answer.

To run against a local task file instead of downloading:

```bash
python examples/arc_agi_demo/scripts/run_demo.py --task-file path/to/task.json
```

## 4. Persist Telemetry (Optional)

Uncomment the `storage` block in `examples/arc_agi_demo/configs/arc_agi_demo.yaml` and provide a PostgreSQL URL (e.g. `postgresql://atlas:atlas@localhost:5432/atlas_arc_demo`).

With persistence enabled you can export traces:

```bash
python examples/arc_agi_demo/scripts/export_traces.py \
  --database-url postgresql://atlas:atlas@localhost:5432/atlas_arc_demo \
  --output arc_agi_traces.jsonl
```

The JSONL file contains the plan, per-step reasoning, reward breakdowns, and the raw trajectory events. This file can be uploaded to Atlas Core or used for custom offline training.

## 5. Upload & Train in Atlas Core

1. Upload the JSONL via the Atlas Core UI or API:
   ```bash
   curl -X POST https://core.arc.computer/api/datasets \
     -H "Authorization: Bearer $ATLAS_CORE_TOKEN" \
     -F "file=@arc_agi_traces.jsonl" \
     -F "name=arc-agi-demo"
   ```
2. Start a training run using the uploaded dataset (UI or API). Monitor progress from the Core dashboard.
3. Invoke the trained model on a new ARC task to showcase reward-guided improvements.

## 6. Cleanup

```bash
deactivate
rm -rf demo-env
```

## Troubleshooting

- **Missing API key** – Ensure `OPENAI_API_KEY` is set before running the scripts.
- **HTTP 404 for task** – Verify the `--task-id` exists in the ARC training split.
- **Postgres errors** – Confirm the URL is reachable and migration permissions are granted.
- **Slow responses** – Switch to a smaller reasoning model by setting `ATLAS_ARC_DEMO_MODEL`.

With these steps you can demonstrate how Atlas orchestrates an existing agent, captures telemetry, and prepares data for further training.
