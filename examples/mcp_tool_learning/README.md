# MCP Tool Learning Example

This example demonstrates how Atlas SDK enables agents to learn efficient tool usage patterns through reinforcement learning. It uses the Model Context Protocol (MCP) to provide file operation tools to a LangGraph agent, showing measurable improvement in tool selection and task completion efficiency over 25 progressive learning runs.

## Overview

**What this demonstrates:**
- MCP server implementation with 5 file system tools
- LangGraph agent integration via `langchain-mcp-adapters`
- Progressive learning across 25 tasks (simple → complex)
- Tool usage optimization through reward signals
- Learning playbook generation and visualization

**Cost estimate:** $1-2 for the complete 25-run learning session using GPT-4.1-mini

## Architecture

```
┌─────────────────────┐
│  learning_harness   │  25 progressive tasks
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Atlas SDK Core    │  Orchestration + reward system
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    mcp_agent.py     │  LangGraph ReAct agent
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ MultiServerMCPClient│  langchain-mcp-adapters
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   mcp_server.py     │  5 file operation tools
└─────────────────────┘
```

## Prerequisites

### 1. Install Dependencies

```bash
pip install arc-atlas langchain-mcp-adapters langchain-openai langgraph mcp anyio
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

Or create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### 3. Start Atlas Infrastructure

```bash
atlas init  # Starts Docker + Postgres for telemetry persistence
```

## Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | MCP server with 5 file tools (read, write, list, search, run_command) |
| `mcp_agent.py` | LangGraph agent that connects to MCP server and exposes tools |
| `config.yaml` | Atlas configuration (Python adapter, LLM settings, reward system) |
| `learning_harness.py` | Runs 25 progressive learning tasks and tracks metrics |
| `sample_workspace/` | Test files for agent operations |

## Quick Start

### Option 1: Run the Full Learning Harness (Recommended)

```bash
cd examples/mcp_tool_learning
python learning_harness.py
```

This executes 25 learning runs with progressive complexity:
1. **Phase 1 (tasks 1-5):** Basic file operations (list, read, write)
2. **Phase 2 (tasks 6-10):** Multi-step operations (copy, search, combine)
3. **Phase 3 (tasks 11-15):** Complex workflows (batch operations, manifests)
4. **Phase 4 (tasks 16-20):** Advanced scenarios (backups, reporting)
5. **Phase 5 (tasks 21-25):** Edge cases and error handling

### Option 2: Run a Single Task

```bash
atlas run --config examples/mcp_tool_learning/config.yaml \
          --task "List all files in sample_workspace and read notes.txt"
```

## MCP Tools Available

The agent has access to 5 MCP tools:

| Tool | Description | Example Use |
|------|-------------|-------------|
| `read_file` | Read file contents | Reading configuration or data files |
| `write_file` | Write/create files | Saving reports or backups |
| `list_files` | List directory contents | Discovering available files |
| `search_content` | Regex search in files | Finding specific patterns or keywords |
| `run_command` | Safe shell commands | Executing ls, grep, wc, etc. |

## Learning Objectives

The agent learns to:
1. **Tool Selection:** Choose the right tool for each operation (e.g., `list_files` before `read_file`)
2. **Efficiency:** Minimize redundant operations (e.g., caching file lists instead of listing repeatedly)
3. **Error Handling:** Gracefully handle missing files and invalid operations
4. **Multi-Step Planning:** Break complex tasks into efficient sequences
5. **Context Awareness:** Understand when to search vs read, list vs execute

## Viewing Learning Progress

### 1. Check Learning Playbook

After running the harness, view the synthesized learning playbook:

```bash
python -m atlas.cli.learning --project mcp-tool-learning
```

This shows:
- Tool usage patterns over time
- Reward progression across sessions
- Common failure modes and recoveries
- Synthesized best practices

### 2. Export Session Traces

```bash
arc-atlas --database-url postgresql://atlas:atlas@localhost:5433/atlas \
          --output mcp_traces.jsonl \
          --limit 25
```

### 3. Query Database Directly

```bash
psql postgresql://atlas:atlas@localhost:5433/atlas

SELECT session_id, task, reward_score, created_at
FROM atlas_sessions
WHERE project_name = 'mcp-tool-learning'
ORDER BY session_id DESC
LIMIT 25;
```

## Expected Results

**Early runs (tasks 1-5):**
- More tool calls per task (trial and error)
- Lower reward scores (~0.6-0.7)
- Occasional incorrect tool selection

**Later runs (tasks 15-25):**
- Fewer tool calls per task (optimized)
- Higher reward scores (~0.8-0.9)
- Consistent correct tool selection
- Better error handling

**Key Metrics:**
- **Tool call reduction:** 30-40% fewer calls in later tasks
- **Completion rate:** 95%+ by task 25
- **Reward progression:** +0.2-0.3 average increase
- **Cost per run:** ~$0.05-0.10 with GPT-4.1-mini

## Understanding the Configuration

The `config.yaml` uses a Python adapter to integrate the MCP agent:

```yaml
agent:
  type: python
  import_path: examples.mcp_tool_learning.mcp_agent
  attribute: create_agent
```

The reward system provides learning signals:

```yaml
rim:
  judge_prompt: |
    Reward effective tool usage:
    - Correct tool for each task
    - Minimal redundant operations
    - Proper error handling
```

## Troubleshooting

### MCP Server Connection Issues

If you see connection errors, verify the MCP server path is correct:
```python
# In mcp_agent.py, check:
server_path = Path(__file__).parent / "mcp_server.py"
```

### Async Event Loop Errors

The agent uses async/await patterns. If you see event loop errors:
```bash
# Ensure you're running with proper async support
python learning_harness.py  # Not: python -i learning_harness.py
```

### API Rate Limits

If you hit rate limits, add delays between tasks:
```python
# In learning_harness.py, increase the sleep duration:
await asyncio.sleep(2)  # Change from 1 to 2 seconds
```

## Next Steps

1. **Customize Tools:** Modify `mcp_server.py` to add domain-specific tools
2. **Adjust Tasks:** Edit `LEARNING_TASKS` in `learning_harness.py` for your use case
3. **Tune Rewards:** Update the `judge_prompt` in `config.yaml` to reward different behaviors
4. **Export for Training:** Use the exported traces to fine-tune your own models

## Cost Breakdown

Approximate costs for the full 25-run learning session:

| Component | Model | Cost per run | Total (25 runs) |
|-----------|-------|--------------|-----------------|
| Student (Agent) | GPT-4.1-mini | ~$0.03 | ~$0.75 |
| Teacher (Validator) | GPT-4.1-mini | ~$0.02 | ~$0.50 |
| Reward system | Gemini-2.5-Flash | ~$0.01 | ~$0.25 |
| **Total** | - | **~$0.06** | **~$1.50** |

**Note:** Actual costs vary based on:
- Task complexity and agent token usage
- Number of tool calls per task
- Retry attempts and error handling
- Reward model evaluation depth

## Related Examples

- `/examples/quickstart.py` - Basic Atlas SDK usage
- `/examples/python_example.py` - Python adapter patterns
- `/deepagents/` - Advanced LangGraph integration with MCP

## License

Apache 2.0 - See main repository LICENSE file
