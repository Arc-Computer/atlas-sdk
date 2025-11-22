# Learning Tracking for BYOA Adapters

This guide explains how to integrate Atlas learning tracking into Bring-Your-Own-Agent (BYOA) adapters.

## Why Learning Tracking Matters

Learning tracking enables Atlas to:
- Measure which playbook entries (tips, insights, warnings) are actually being used
- Track when specific tools are adopted based on learning recommendations
- Compute learning effectiveness metrics across sessions
- Generate impact reports showing which learning improves performance

Without learning tracking, you can't measure whether your agent is actually learning from experience.

## Quick Start

Here's the minimal code to enable learning tracking in a BYOA adapter:

```python
from typing import Any, Dict
from atlas.learning.playbook import resolve_playbook
from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext

def my_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """BYOA adapter with learning tracking."""
    metadata = metadata or {}
    context = ExecutionContext.get()
    tracker = get_tracker(context)

    # 1. Retrieve playbook entries (automatically registers them)
    playbook, digest, playbook_metadata = resolve_playbook("student", apply=True)

    # 2. Detect cue hits from user input
    user_input = metadata.get("task_payload", prompt)
    tracker.detect_and_record("student", user_input, step_id=1)

    # 3. Execute your agent logic
    result = execute_agent(prompt, metadata)

    # 4. Track tool adoptions
    if result.get("tool_name"):
        tracker.record_action_adoption(
            "student",
            runtime_handle=result["tool_name"],
            success=True,
            step_id=1
        )

    # 5. Record session outcome
    tracker.record_session_outcome(
        reward_score=0.8,
        token_usage={"total_tokens": 1500}
    )

    return result
```

That's it. These five calls enable full learning tracking.

## Core Concepts

### Playbook Entries

Playbook entries are learning insights generated from successful sessions. Each entry has:
- **Cue**: Pattern that triggers the learning (keyword, regex, predicate)
- **Action**: What to do when cue is detected (often a `runtime_handle` like a tool name)
- **Expected Effect**: Why this action helps

When you call `resolve_playbook()`, entries are **automatically registered** with the tracker.

### Cue Detection

Cues are patterns in user input that trigger learning. Types:
- `keyword`: Simple substring match
- `regex`: Regular expression pattern
- `predicate`: Condition-based match

When a cue matches user input, it's counted as a "cue hit."

### Action Adoption

When your agent executes a tool that matches a playbook entry's `runtime_handle`, it's counted as an "action adoption." This tracks whether learning actually changes behavior.

### Learning Keys

Learning keys group sessions that should share learning state. Common strategies:
- **Per-project**: All sessions in a project learn together
- **Per-task-type**: Group by task category
- **Fixed evaluation key**: Use same key across restarts for research

## API Reference

### resolve_playbook()

Retrieves learning state for Student or Teacher persona.

```python
from atlas.learning.playbook import resolve_playbook

playbook, digest, metadata = resolve_playbook(
    role="student",  # or "teacher"
    apply=True       # Set False to disable learning
)
```

**Returns:**
- `playbook` (str): Text to inject into prompts
- `digest` (str): SHA-256 hash for caching
- `metadata` (dict): Contains `playbook_entries` list

**Important:** Calling this function **automatically registers** playbook entries with the tracker. You don't need to call `tracker.register_entries()` manually.

**Location:** `atlas/learning/playbook.py:15-102`

---

### get_tracker()

Gets the learning usage tracker for current execution context.

```python
from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext

context = ExecutionContext.get()
tracker = get_tracker(context)
```

**Location:** `atlas/learning/usage.py:247-251`

---

### tracker.detect_and_record()

Detects and records cue hits from text.

```python
matches = tracker.detect_and_record(
    role="student",           # "student" or "teacher"
    text="user input text",   # Text to scan for cues
    step_id=1,                # Optional: step identifier
    context_hint="snippet"    # Optional: snippet to save as example
)
```

**Returns:** List of matched cue patterns

**When to call:** Whenever you process user input or text that might contain learning cues.

**Location:** `atlas/learning/usage.py:91-112`

---

### tracker.record_action_adoption()

Records when a tool/action is adopted.

```python
tracker.record_action_adoption(
    role="student",
    runtime_handle="tool_name",  # Must match playbook entry's runtime_handle
    success=True,                # Whether action succeeded
    step_id=1,                   # Optional: step identifier
    metadata={"extra": "data"}   # Optional: additional context
)
```

**Critical:** The `runtime_handle` must **exactly match** the `action.runtime_handle` in playbook entries. This is how Atlas knows which learning entry was adopted.

**When to call:** After executing any tool or action that might be in playbook entries.

**Location:** `atlas/learning/usage.py:141-177`

---

### tracker.record_session_outcome()

Records final session metrics.

```python
tracker.record_session_outcome(
    reward_score=0.85,
    token_usage={"total_tokens": 2000, "calls": 5},
    incident_id="incident-123",
    task_identifier="security-review",
    incident_tags=["security", "authentication"],
    retry_count=2,
    failure_flag=False
)
```

**When to call:** At the end of session execution, before returning results.

**Location:** `atlas/learning/usage.py:178-239`

---

## Complete Working Example

Here's a full BYOA adapter with learning tracking for a CRM-style agent:

```python
import json
from typing import Any, Dict
from atlas.learning.playbook import resolve_playbook
from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext


def crm_agent_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    CRM agent with learning tracking.

    Tracks:
    - Cue detection from user utterances
    - Tool adoption when executing CRM operations
    - Session outcome with success metrics
    """
    metadata = metadata or {}
    mode = metadata.get("mode")

    # Get execution context and tracker
    try:
        context = ExecutionContext.get()
        tracker = get_tracker(context)
        tracking_enabled = True
    except Exception:
        # Graceful degradation if context unavailable (e.g., testing)
        tracking_enabled = False

    # Retrieve learning state (automatically registers entries)
    if tracking_enabled:
        playbook, digest, playbook_meta = resolve_playbook("student", apply=True)
        if playbook:
            print(f"Using learning playbook (digest: {digest[:8]}...)")

    # PLANNING PHASE
    if mode == "planning":
        task = metadata.get("task_payload", "")

        # Detect cues in task description
        if tracking_enabled:
            tracker.detect_and_record("student", task, step_id=0)

        # Generate plan (your logic here)
        plan = generate_plan(task)
        return json.dumps(plan)

    # EXECUTION PHASE
    step_payload = metadata.get("step_payload", {})
    if step_payload:
        step_id = step_payload.get("step_id", 1)
        user_utterance = step_payload.get("description", "")

        # Detect cues in user utterance
        if tracking_enabled:
            matches = tracker.detect_and_record(
                "student",
                user_utterance,
                step_id=step_id,
                context_hint=user_utterance[:200]
            )
            if matches:
                print(f"Detected {len(matches)} cue hits in step {step_id}")

        # Execute CRM operation (your logic here)
        result = execute_crm_operation(user_utterance, step_payload)

        # Track tool adoption if tool was used
        if tracking_enabled and result.get("tool_name"):
            tool_name = result["tool_name"]
            success = result.get("success", True)

            tracker.record_action_adoption(
                "student",
                runtime_handle=tool_name,
                success=success,
                step_id=step_id,
                metadata={
                    "tool_name": tool_name,
                    "arguments": result.get("arguments", {}),
                    "execution_time": result.get("execution_time_ms")
                }
            )

            print(f"Tracked adoption of '{tool_name}' (success={success})")

        return json.dumps(result)

    # Fallback
    return json.dumps({"error": "No structured payload available"})


def generate_plan(task: str) -> Dict:
    """Generate execution plan from task."""
    # Your planning logic here
    return {
        "steps": [
            {"id": 1, "description": f"Process: {task}", "depends_on": []}
        ]
    }


def execute_crm_operation(utterance: str, step_payload: Dict) -> Dict:
    """Execute CRM operation and return result."""
    # Your execution logic here
    # This would call actual CRM tools, parse results, etc.
    return {
        "tool_name": "create_contact",
        "arguments": {"name": "John Doe"},
        "success": True,
        "execution_time_ms": 150,
        "result": "Contact created successfully"
    }
```

## Learning Key Management

### Default Behavior

By default, Atlas generates learning keys from:
- Config parameters
- Task metadata
- Timestamp (which causes new key on each restart)

### Fixed Learning Keys for Evaluation

For evaluation scenarios that span multiple restarts, use a fixed learning key:

```python
import hashlib

# In your adapter or test harness
learning_key = hashlib.sha256(
    "my-evaluation-2025-11-22".encode("utf-8")
).hexdigest()

# Store in ExecutionContext metadata
context = ExecutionContext.get()
context.metadata["learning_key"] = learning_key
```

This ensures all sessions share the same learning state across restarts.

### Best Practices

- **Single project:** Use project name as learning key
- **Multiple task types:** Include task type in key (e.g., `project-tasktype`)
- **A/B testing:** Use different keys for control vs experimental groups
- **Evaluation runs:** Use fixed key with date/version identifier

## Troubleshooting

### Adoptions Show 0 Despite Tool Execution

**Problem:** Tools are executing but `action_adoptions` remains 0.

**Cause:** The `runtime_handle` in `record_action_adoption()` doesn't match playbook entries.

**Solution:**
```python
# Check what runtime_handles are registered
playbook, digest, metadata = resolve_playbook("student", apply=True)
if metadata:
    for entry in metadata.get("playbook_entries", []):
        print(f"Entry {entry['id']}: runtime_handle={entry.get('action', {}).get('runtime_handle')}")

# Ensure your tool_name matches exactly
tracker.record_action_adoption("student", runtime_handle="exact_tool_name", success=True)
```

---

### Cue Hits Not Being Detected

**Problem:** No cue hits recorded even when patterns should match.

**Cause:** Cue patterns in playbook entries may be incorrect, or text not being scanned.

**Solution:**
```python
# Debug cue detection
playbook, digest, metadata = resolve_playbook("student", apply=True)
if metadata:
    for entry in metadata.get("playbook_entries", []):
        cue = entry.get("cue", {})
        print(f"Cue type={cue.get('type')}, pattern={cue.get('pattern')}")

# Test detection manually
text = "user input text"
matches = tracker.detect_and_record("student", text, step_id=1)
print(f"Matched {len(matches)} cues: {matches}")
```

---

### Learning State Not Persisting

**Problem:** Learning resets on each run.

**Cause:** No database configured, or learning key changes each run.

**Solution:**
1. Ensure Postgres is configured in your config:
   ```yaml
   storage:
     database_url: postgresql://atlas:atlas@localhost:5433/atlas
   ```

2. Use fixed learning key for evaluation scenarios (see above).

---

### Playbook Entries Not Registered

**Problem:** `playbook_entries` list is empty.

**Cause:** Either no learning has been synthesized yet, or `resolve_playbook()` not being called.

**Solution:**
1. Run several high-reward sessions to trigger learning synthesis
2. Verify learning synthesis is enabled in config:
   ```yaml
   learning:
     enabled: true
     update_enabled: true
   ```

3. Ensure you're calling `resolve_playbook("student", apply=True)` with `apply=True`

---

## Integration Checklist

Use this checklist when adding learning tracking to a BYOA adapter:

- [ ] Call `resolve_playbook("student", apply=True)` to retrieve and register playbook entries
- [ ] Call `tracker.detect_and_record()` on user input/task text
- [ ] Call `tracker.record_action_adoption()` after tool execution with matching `runtime_handle`
- [ ] Call `tracker.record_session_outcome()` with final metrics
- [ ] Configure Postgres database for persistence
- [ ] Use fixed learning key for evaluation scenarios (if applicable)
- [ ] Test that cue hits and adoptions are being recorded (check database or logs)
- [ ] Verify learning synthesis is enabled in config

## Related Documentation

- [Structured Adapter Payloads](structured_adapter_payloads.md) - BYOA adapter integration guide
- [Configuration Guide](../configs/configuration.md) - Learning system configuration
- [Introduction](../guides/introduction.mdx) - Getting started with Atlas SDK

## Code References

- Learning playbook: `atlas/learning/playbook.py`
- Usage tracker: `atlas/learning/usage.py`
- Execution context: `atlas/runtime/orchestration/execution_context.py`
- Example adapter: `examples/byoa_structured_adapter.py`
