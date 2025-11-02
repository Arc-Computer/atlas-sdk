# Terminal Telemetry Walkthrough

Atlas streams orchestration events directly to stdout, so you can follow progress without a browser.

## Prerequisites

- Install the SDK and dev extras:

  ```bash
  python3.13 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .[dev]
  ```

- Configure the desired example, such as `examples/mcp_tool_learning/config.yaml`.

## Run a Task with Streaming

```bash
python - <<'PY'
from atlas import core

result = core.run(
    task="Summarize the Atlas SDK",
    config_path="examples/mcp_tool_learning/config.yaml",
)

print(result.final_answer)
PY
```

When `stdout` is a TTY, the console renderer activates automatically. Sample output:

```text
=== Atlas task started: Summarize the Atlas SDK (2025-02-12 10:15:03) ===
Plan ready with steps:
  1. gather dataset A
  2. synthesise findings
[step 1] attempt 1 started: gather dataset A
[tool] web_search call -> {"query": "Atlas SDK release"}
[tool] web_search result <- {"result": "..."}
[step 1] completed: gather dataset A
  reward score: 0.91
[step 2] retry 2 started: synthesise findings
  guidance: cite the repository README
=== Atlas task completed in 12.4s ===
Final answer:
  Atlas SDK ships a teacher-student loop...
- gather dataset A | attempts: 1 | score: 0.91
- synthesise findings | attempts: 2 | score: 0.88
Reward-system scores | max: 0.91 | avg: 0.89
```

- Disable streaming: set `stream_progress=False` when calling `core.run` or `core.arun`.
- Force streaming when redirecting output: set `stream_progress=True`.
- Persistence remains available; database logging still captures the same events for audit and replay.
