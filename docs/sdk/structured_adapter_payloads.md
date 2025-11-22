# Structured Task and Step Payloads for BYOA Adapters

## Overview

Atlas SDK supports passing structured task and step payloads to BYOA (Bring-Your-Own-Agent) adapters through the `metadata` parameter. This **opt-in** feature enables integration with test harnesses, simulation environments, and other structured agent scenarios where deterministic execution is needed.

**Important:** Structured payloads are **only sent to adapters that explicitly opt-in** by setting `supports_structured_payloads = True`. LLM-based adapters (OpenAI, LiteLLM, Anthropic) keep this flag `False` by default to prevent leaking structured data to external providers.

## What's Available

### Opt-In Requirement

To receive structured payloads, your adapter must set:

```python
class MyBYOAAdapter(AgentAdapter):
    supports_structured_payloads = True  # Enable structured payloads
```

Without this flag, the bridge will **not** enrich metadata with `task_payload` or `step_payload`.

### Metadata Fields

Adapters that opt-in receive metadata through the `ainvoke(prompt: str, metadata: Dict[str, Any] | None)` method with the following structured fields:

#### Planning Phase
```python
metadata = {
    "mode": "planning",
    "task_payload": str  # Original task before prompt composition
}
```

#### Execution Phase
```python
metadata = {
    "mode": None,  # Not explicitly set during execution
    "task_payload": str,  # Original task
    "step_payload": {
        "step_id": int,
        "description": str,
        "depends_on": List[int]
    },
    "tools": List[Dict],  # Available tool definitions (OpenAI format)
    "tool_choice": str  # Tool selection strategy
}
```

### Payload Size

Structured payloads are lightweight (~500 bytes - 2KB per step) to avoid:
- Exceeding LLM context windows during learning synthesis
- Database bloat in session storage
- Network transfer overhead

## Use Cases

### 1. Test Harness Integration

Perfect for integrating test frameworks like the Arc CRM benchmark:

```python
from atlas.connectors.registry import AgentAdapter

class CRMHarnessAdapter(AgentAdapter):
    """Adapter that executes against a deterministic test harness."""

    supports_structured_payloads = True  # Opt-in to receive structured payloads

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        metadata = metadata or {}

        task = metadata.get("task_payload")
        step = metadata.get("step_payload")

        if metadata.get("mode") == "planning":
            # Load test scenario from dataset using task
            conversation = load_conversation_from_dataset(task)
            return json.dumps({
                "steps": [
                    {"id": turn.turn_id, "description": turn.user_utterance, "depends_on": []}
                    for turn in conversation.turns
                ]
            })

        if step:
            # Reconstruct turn context from step payload
            turn_id = step["step_id"]
            conversation = load_conversation_from_dataset(task)
            turn = conversation.turns[turn_id - 1]

            # Execute ground truth tool call
            context = AgentTurnContext(
                conversation=conversation,
                turn=turn,
                prior_turns=conversation.turns[:turn_id-1],
                previous_results={},
                expected_arguments=turn.expected_args
            )

            result = harness.execute_turn(context)
            return json.dumps(result)
```

### 2. Simulation Environments

For replay-based or dataset-driven agents:

```python
def simulation_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Adapter that replays predetermined actions from a simulation."""
    step = metadata.get("step_payload") if metadata else None

    if step:
        # Load predetermined action for this step
        action = simulation_db.get_action(step["step_id"])
        return json.dumps({
            "tool_name": action.tool,
            "arguments": action.args
        })

    # Fallback to prompt-based response
    return llm_completion(prompt)
```

### 3. Hybrid Approaches

Combine structured and LLM-based execution:

```python
async def hybrid_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Use structured data when available, LLM when needed."""
    metadata = metadata or {}

    task = metadata.get("task_payload")
    step = metadata.get("step_payload")

    # Try structured execution first
    if task and step:
        cached_plan = plan_cache.get(task)
        if cached_plan and step["step_id"] <= len(cached_plan):
            # Execute cached action
            action = cached_plan[step["step_id"] - 1]
            return execute_action(action)

    # Fall back to LLM
    return await llm.ainvoke(prompt, metadata=metadata)
```

## Backward Compatibility

### Existing Adapters Continue to Work

All existing adapters automatically benefit from graceful degradation:

**LiteLLM/OpenAI Adapters:**
- Ignore unknown metadata keys
- Continue to extract only `tools`, `tool_choice`, and `messages`

**Python Adapters:**
- Receive full metadata dict
- Can optionally check for `task_payload` and `step_payload`
- Functions that don't use these fields work unchanged

**Example - Legacy adapter:**
```python
def legacy_adapter(prompt: str, metadata: Dict = None):
    # Ignores metadata completely - still works!
    return process_prompt(prompt)
```

**Example - Opportunistic usage:**
```python
def smart_adapter(prompt: str, metadata: Dict = None):
    # Uses structured data when available, falls back to prompt
    if metadata and metadata.get("step_payload"):
        step_id = metadata["step_payload"]["step_id"]
        return f"Executing step {step_id}"
    return process_prompt(prompt)
```

## Implementation Details

### How It Works

1. **Planning Phase** (Student.aplan_task):
   - Student stores task in ExecutionContext
   - Calls adapter with `{"mode": "planning", "task_payload": task}`

2. **Execution Phase** (Student.aexecute_step):
   - Student stores lightweight step payload in ExecutionContext
   - LangChain bridge opportunistically enriches metadata
   - Adapter receives `{"task_payload": ..., "step_payload": {...}}`

3. **Graceful Degradation**:
   - If ExecutionContext unavailable → metadata not enriched
   - Adapter still called with basic metadata
   - No errors, just missing optional fields

### Code References

- Student persona: `atlas/personas/student.py:183-187, 243-248`
- LangChain bridge: `atlas/connectors/langchain_bridge.py:239-256`
- Example adapter: `examples/byoa_structured_adapter.py`
- Tests: `tests/unit/test_structured_payloads.py`

## Best Practices

### DO: Keep Payloads Lightweight

```python
# GOOD - Lightweight reference data
step_payload = {
    "step_id": 1,
    "description": "Create customer",
    "depends_on": []
}
# Size: ~100 bytes
```

### DON'T: Include Large Execution State

```python
# BAD - Bloats metadata
metadata["full_execution_trace"] = {
    "all_previous_outputs": [...],  # 50KB
    "complete_conversation_history": [...],  # 100KB
    "raw_llm_responses": {...}  # 20KB per step
}
# Size: 170KB+ per step → serialization overhead, context window issues
```

### DO: Check for Availability

```python
def robust_adapter(prompt: str, metadata: Dict = None):
    metadata = metadata or {}

    # Safely check for structured payloads
    task = metadata.get("task_payload")
    step = metadata.get("step_payload")

    if task and step:
        # Use structured execution
        return execute_structured(task, step)

    # Fallback to prompt-based
    return execute_prompt(prompt)
```

### DON'T: Assume Always Present

```python
# BAD - Will crash if metadata not enriched
def brittle_adapter(prompt: str, metadata: Dict):
    step_id = metadata["step_payload"]["step_id"]  # KeyError if missing!
    return execute(step_id)
```

## Performance Considerations

### Payload Size Impact

| Component | Impact | Safe Threshold |
|-----------|--------|----------------|
| Serialization | +10-20% with 1-2KB payloads | < 2KB per step |
| LLM Context (synthesis) | Risk of overflow with large payloads | < 1KB per step |
| Database Storage | JSON column growth | < 5KB per step |
| Network Transfer | Latency increase | < 2KB per step |

### Monitoring

Monitor these metrics if using structured payloads extensively:

```python
# In your adapter
import sys

payload_size = sys.getsizeof(json.dumps(metadata))
if payload_size > 2048:  # 2KB
    logger.warning(f"Large metadata payload: {payload_size} bytes")
```

## Troubleshooting

### Payload Not Available

**Symptom:** `metadata.get("task_payload")` returns `None`

**Causes:**
1. ExecutionContext not initialized (e.g., standalone adapter testing)
2. Adapter called outside Atlas runtime
3. Exception during context enrichment

**Solution:** Always provide fallback behavior:
```python
task = metadata.get("task_payload")
if not task:
    # Fall back to parsing prompt or default behavior
    task = extract_task_from_prompt(prompt)
```

### Payload Too Large

**Symptom:** Slow performance, database timeouts, context window errors

**Causes:**
1. Including full execution traces in metadata
2. Storing complete conversation histories
3. Not filtering unnecessary data

**Solution:** Only include essential reference data:
```python
# Instead of full context, pass minimal identifiers
step_payload = {
    "step_id": step.id,  # Use to lookup full context if needed
    "description": step.description[:100],  # Truncate long descriptions
    "depends_on": step.depends_on
}
```

## Migration Guide

### Updating Existing Adapters

No migration required! Existing adapters continue to work. To opt-in to structured payloads:

**Before:**
```python
def my_adapter(prompt: str, metadata: Dict = None):
    # Parse prompt to extract task details
    task_match = re.search(r"Task: (.+)", prompt)
    task = task_match.group(1) if task_match else "unknown"
    return execute(task)
```

**After:**
```python
def my_adapter(prompt: str, metadata: Dict = None):
    metadata = metadata or {}

    # Use structured payload when available
    task = metadata.get("task_payload")

    # Fall back to prompt parsing
    if not task:
        task_match = re.search(r"Task: (.+)", prompt)
        task = task_match.group(1) if task_match else "unknown"

    return execute(task)
```

## Examples

See `examples/byoa_structured_adapter.py` for complete working examples including:
- Basic structured adapter
- Test harness integration pattern
- Hybrid LLM/structured approach

## Learning Tracking Integration

BYOA adapters can integrate with Atlas learning tracking to measure which playbook entries are used and which actions are adopted during execution.

**Key Benefits:**
- Measure learning effectiveness across sessions
- Track which tools are adopted based on learning recommendations
- Compute impact metrics for playbook entries
- Enable research/evaluation scenarios with learning accumulation

**See:** [Learning Tracking for BYOA Adapters](learning_tracking.md) for complete integration guide.

**Quick Example:**
```python
from atlas.learning.playbook import resolve_playbook
from atlas.learning.usage import get_tracker

def my_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    # 1. Retrieve playbook (auto-registers entries)
    playbook, digest, meta = resolve_playbook("student", apply=True)

    # 2. Get tracker
    tracker = get_tracker()

    # 3. Detect cue hits from user input
    tracker.detect_and_record("student", user_input, step_id=1)

    # 4. Execute your logic...
    result = execute(prompt, metadata)

    # 5. Track tool adoptions
    tracker.record_action_adoption("student", tool_name, success=True, step_id=1)

    # 6. Record session outcome
    tracker.record_session_outcome(reward_score=0.85)

    return result
```

---

## Further Reading

- **[Learning Tracking for BYOA Adapters](learning_tracking.md)** - Complete learning integration guide
- [BYOA Configuration Guide](../configs/configuration.md)
- [Agent Patterns](agent_patterns.md)
- [Arc CRM Integration Example](../../examples/arc_crm_integration/) (coming soon)
