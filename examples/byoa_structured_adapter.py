"""
Example BYOA adapter demonstrating structured task/step payload usage.

This example shows how to build a deterministic adapter (like a test harness)
that uses structured task and step payloads instead of parsing natural language prompts.

Use case: Integration with test frameworks, simulation environments, or any
BYOA scenario where the "agent" needs access to structured execution context.

Usage:
    Create a config.yaml with:

    adapter:
      type: python
      import_path: examples.byoa_structured_adapter
      attribute: structured_adapter

    Then run:
    atlas run --task "your task here"

Examples in this file:
    1. structured_adapter - Basic structured payload usage
    2. harness_adapter_example - Test harness integration pattern
    3. learning_enabled_adapter - Complete learning tracking integration

For learning tracking guide, see: docs/sdk/learning_tracking.md
"""

from typing import Any, Dict
import json


def structured_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    Example adapter that demonstrates access to structured payloads.

    This adapter receives:
    - prompt: Composed natural language prompt (can be ignored for structured integrations)
    - metadata: Dict containing:
        - mode: "planning" | "execution" (phase indicator)
        - task_payload: Original task string (available in all phases)
        - step_payload: Dict with step_id, description, depends_on (execution phase only)
        - tools: List of available tools (execution phase only)
        - tool_choice: Tool selection strategy (execution phase only)

    Returns:
        str: Response (for planning: JSON plan, for execution: step output)
    """
    metadata = metadata or {}

    # Extract mode to determine phase
    mode = metadata.get("mode", "unknown")

    # Extract structured payloads
    task_payload = metadata.get("task_payload")
    step_payload = metadata.get("step_payload")

    print(f"\n=== Structured Adapter Called ===")
    print(f"Mode: {mode}")
    print(f"Task: {task_payload}")

    if mode == "planning":
        # Planning phase: Return a simple plan
        # In a real integration, you might:
        # - Parse task_payload to extract requirements
        # - Generate plan from test scenarios
        # - Load plan from configuration

        print("Generating plan from structured task...")
        plan = {
            "steps": [
                {
                    "id": 1,
                    "description": f"Execute task: {task_payload}",
                    "depends_on": []
                }
            ]
        }
        return json.dumps(plan)

    elif mode == "execution" or step_payload:
        # Execution phase: Access structured step context
        # In a real integration (like Arc CRM), you would:
        # - Reconstruct turn context from step_payload
        # - Execute ground-truth tool calls
        # - Validate against expected results

        if step_payload:
            step_id = step_payload.get("step_id")
            description = step_payload.get("description")
            depends_on = step_payload.get("depends_on", [])

            print(f"Step {step_id}: {description}")
            print(f"Dependencies: {depends_on}")

        # Access tool definitions if needed
        tools = metadata.get("tools", [])
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            print(f"Available tools: {tool_names}")

        # Example: Return structured output with tool call
        # For harness integrations, this would be the ground-truth tool call
        output = {
            "tool_name": "example_tool",
            "arguments": {"task": task_payload},
            "result": f"Executed step {step_payload.get('step_id') if step_payload else 'unknown'}"
        }
        return json.dumps(output)

    else:
        # Fallback for unknown modes
        print("Unknown mode, falling back to prompt-based response")
        return f"Processed: {prompt}"


def harness_adapter_example(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    More advanced example showing test harness integration pattern.

    This demonstrates how Arc CRM benchmark could integrate with Atlas:
    1. Receive structured task and step payloads
    2. Reconstruct execution context
    3. Execute deterministic tool calls from ground truth
    4. Return validation results
    """
    metadata = metadata or {}

    # For harness integrations, you typically:
    # 1. Load ground truth from dataset using task_payload
    # 2. Use step_payload to identify which turn to execute
    # 3. Execute against mock/sandbox environment
    # 4. Return validation results

    task = metadata.get("task_payload")
    step = metadata.get("step_payload")
    mode = metadata.get("mode")

    if mode == "planning":
        # Load test scenario steps from dataset
        # For this example, we'll use a simple hardcoded plan
        return json.dumps({
            "steps": [
                {"id": 1, "description": "Setup test environment", "depends_on": []},
                {"id": 2, "description": "Execute test case", "depends_on": [1]},
                {"id": 3, "description": "Validate results", "depends_on": [2]}
            ]
        })

    if step:
        # This is where you'd integrate with your harness
        # Example flow for Arc CRM:
        #
        # 1. Get conversation from dataset using task_payload
        # 2. Get turn from conversation using step["step_id"]
        # 3. Build AgentTurnContext with:
        #    - conversation metadata
        #    - current turn
        #    - prior turns
        #    - previous results
        # 4. Call harness.execute_turn(context)
        # 5. Return validation result

        step_id = step.get("step_id", 0)

        # Simulated harness execution
        result = {
            "turn_id": step_id,
            "tool_call": {
                "tool": "ground_truth_tool",
                "arguments": {"from_dataset": True}
            },
            "validation": {"passed": True, "score": 1.0},
            "trace": f"Executed ground truth for step {step_id}"
        }
        return json.dumps(result)

    return "No structured payload available"


def learning_enabled_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    Example BYOA adapter with complete learning tracking integration.

    Demonstrates:
    - Automatic playbook retrieval and entry registration
    - Cue hit detection from user input
    - Action adoption tracking after tool execution
    - Session outcome recording

    This example shows the five core learning tracking calls:
    1. get_tracker() - Accesses the tracker from ExecutionContext
    2. resolve_playbook() - Retrieves and auto-registers playbook entries
    3. detect_and_record() - Detects cue hits from text
    4. record_action_adoption() - Tracks when tools are adopted
    5. record_session_outcome() - Records final metrics

    See docs/sdk/learning_tracking.md for detailed guide.

    Usage:
        adapter:
          type: python
          import_path: examples.byoa_structured_adapter
          attribute: learning_enabled_adapter
    """
    metadata = metadata or {}

    # Import learning tracking modules (placed here for clarity)
    try:
        from atlas.learning.playbook import resolve_playbook
        from atlas.learning.usage import get_tracker
        from atlas.runtime.orchestration.execution_context import ExecutionContext
    except ImportError:
        # Graceful degradation if imports fail
        return _fallback_execution(prompt, metadata)

    # Get execution context and tracker
    try:
        context = ExecutionContext.get()
        tracker = get_tracker(context)
        tracking_enabled = True
    except Exception:
        # Context not available (e.g., standalone testing)
        tracking_enabled = False
        print("⚠️  ExecutionContext unavailable - learning tracking disabled")

    # Step 1: Retrieve learning playbook (automatically registers entries)
    if tracking_enabled:
        try:
            playbook, digest, playbook_meta = resolve_playbook("student", apply=True)
            if playbook:
                print(f"✓ Retrieved playbook (digest: {digest[:8] if digest else 'none'}...)")
                if playbook_meta:
                    entry_count = len(playbook_meta.get("playbook_entries", []))
                    print(f"  └─ Registered {entry_count} playbook entries")
        except Exception as e:
            print(f"⚠️  Playbook retrieval failed: {e}")
            playbook_meta = None

    # Extract mode and payloads
    mode = metadata.get("mode")
    task_payload = metadata.get("task_payload", "")
    step_payload = metadata.get("step_payload")

    # PLANNING PHASE
    if mode == "planning":
        # Step 2: Detect cues in task description
        if tracking_enabled and task_payload:
            try:
                matches = tracker.detect_and_record(
                    role="student",
                    text=task_payload,
                    step_id=0,  # Planning phase is step 0
                    context_hint=task_payload[:200]
                )
                if matches:
                    print(f"✓ Detected {len(matches)} cue hits in task description")
            except Exception as e:
                print(f"⚠️  Cue detection failed: {e}")

        # Generate plan
        plan = {
            "steps": [
                {
                    "id": 1,
                    "description": f"Execute task: {task_payload}",
                    "depends_on": []
                }
            ]
        }
        return json.dumps(plan)

    # EXECUTION PHASE
    if step_payload:
        step_id = step_payload.get("step_id", 1)
        description = step_payload.get("description", "")

        print(f"\n=== Executing Step {step_id} ===")

        # Step 2: Detect cues in step description
        if tracking_enabled and description:
            try:
                matches = tracker.detect_and_record(
                    role="student",
                    text=description,
                    step_id=step_id,
                    context_hint=description[:200]
                )
                if matches:
                    print(f"✓ Detected {len(matches)} cue hits in step description")
                    for match in matches:
                        print(f"  └─ Cue type: {match.get('type')}, pattern: {match.get('pattern')}")
            except Exception as e:
                print(f"⚠️  Cue detection failed: {e}")

        # Execute the step (simulate tool execution)
        tool_result = _simulate_tool_execution(description, step_id)

        # Step 3: Record action adoption if tool was used
        if tracking_enabled and tool_result.get("tool_name"):
            tool_name = tool_result["tool_name"]
            success = tool_result.get("success", True)

            try:
                tracker.record_action_adoption(
                    role="student",
                    runtime_handle=tool_name,  # Must match playbook entry's runtime_handle
                    success=success,
                    step_id=step_id,
                    metadata={
                        "tool_name": tool_name,
                        "arguments": tool_result.get("arguments", {}),
                        "execution_time_ms": tool_result.get("execution_time_ms", 0)
                    }
                )
                print(f"✓ Tracked adoption of '{tool_name}' (success={success})")
            except Exception as e:
                print(f"⚠️  Action adoption tracking failed: {e}")

        # Step 4: Record session outcome (typically done once at end)
        # For multi-step sessions, you'd accumulate metrics and call this once
        if tracking_enabled and step_id == 1:  # Only on final step in real usage
            try:
                tracker.record_session_outcome(
                    reward_score=0.85,  # From your evaluation logic
                    token_usage={
                        "total_tokens": 1500,
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "calls": 1
                    },
                    task_identifier=f"step-{step_id}",
                    retry_count=0,
                    failure_flag=False
                )
                print(f"✓ Recorded session outcome (reward: 0.85)")
            except Exception as e:
                print(f"⚠️  Session outcome recording failed: {e}")

        return json.dumps(tool_result)

    # Fallback for unknown mode
    return _fallback_execution(prompt, metadata)


def _simulate_tool_execution(description: str, step_id: int) -> Dict[str, Any]:
    """
    Simulate tool execution for demonstration purposes.

    In a real adapter, this would:
    - Parse the description to determine which tool to call
    - Execute the actual tool with appropriate arguments
    - Return structured results with success status
    """
    # Simple simulation: return a mock tool execution result
    return {
        "tool_name": "execute_task",  # This should match playbook runtime_handle
        "arguments": {
            "task": description,
            "step_id": step_id
        },
        "success": True,
        "execution_time_ms": 150,
        "result": f"Successfully executed step {step_id}",
        "trace": f"Processed description: {description}"
    }


def _fallback_execution(prompt: str, metadata: Dict[str, Any]) -> str:
    """Fallback execution when learning tracking is unavailable."""
    return json.dumps({
        "status": "executed",
        "message": "Fallback execution (learning tracking unavailable)",
        "prompt_received": prompt[:100]
    })


# Export the adapter functions
__all__ = ["structured_adapter", "harness_adapter_example", "learning_enabled_adapter"]
