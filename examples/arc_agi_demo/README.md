# ARC-AGI Bring-Your-Own-Agent Demo

This example illustrates how to wrap an existing OpenAI-powered agent with Atlas using only the published `arc-atlas` package. It includes:

- `agent/demo_agent.py` – async wrapper around the OpenAI Responses API.
- `configs/arc_agi_demo.yaml` – Atlas configuration that wires the agent into the Teacher → Student → Reward loop.
- `scripts/run_baseline.py` – calls the OpenAI agent directly to establish a baseline answer.
- `scripts/run_demo.py` – downloads (or loads locally) an ARC task and executes the Atlas orchestration pipeline.
- `scripts/export_traces.py` – exports persisted telemetry to JSONL (requires Postgres).
- `docs/arc_agi_demo_walkthrough.md` – step-by-step setup and training instructions.

Follow the walkthrough to set up a virtual environment, run the demo, export traces, and upload data to Atlas Core. All commands work against a plain `pip install arc-atlas` installation.
