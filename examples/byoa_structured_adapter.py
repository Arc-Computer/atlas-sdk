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
"""

from typing import Any, Dict


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
        import json
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
        import json
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
        import json
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
        import json
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


# Export the adapter function
__all__ = ["structured_adapter", "harness_adapter_example"]
