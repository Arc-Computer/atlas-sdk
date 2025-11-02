#!/usr/bin/env python3
"""Validation script for tool-backed adoption tracking.

This script validates that tool adoption tracking works end-to-end with agentic adapters:
1. Runs a task that triggers tool calls via an agentic adapter (e.g., MCP)
2. Provides playbook entries via test_learning_state parameter
3. Verifies cue hits and adoption tracking in ExecutionContext metadata
4. Verifies runtime_handles are populated from tool calls

Usage:
    python scripts/validate_tool_adoption.py --config configs/eval/learning/tool_adoption_openai.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.core import arun as atlas_arun
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.env import load_dotenv_if_available


# Playbook entries matching tool names from tool_adoption configs
PLAYBOOK_ENTRIES = [
    {
        "id": "web_search_entry",
        "cue": {"type": "keyword", "pattern": "search|lookup|find"},
        "action": {"imperative": "Use web_search tool", "runtime_handle": "web_search"},
        "scope": {"category": "reinforcement"},
    },
    {
        "id": "calculate_entry",
        "cue": {"type": "keyword", "pattern": "calculate|compute|sum|add"},
        "action": {"imperative": "Use calculate tool", "runtime_handle": "calculate"},
        "scope": {"category": "reinforcement"},
    },
    {
        "id": "format_text_entry",
        "cue": {"type": "keyword", "pattern": "format|summary|report"},
        "action": {"imperative": "Use format_text tool", "runtime_handle": "format_text"},
        "scope": {"category": "reinforcement"},
    },
]


async def validate_tool_adoption(config_path: str) -> tuple[bool, str]:
    """Run a task and validate tool adoption tracking with agentic adapter."""
    load_dotenv_if_available()
    
    # Task that should trigger tool calls
    task = (
        "Search for information about Atlas SDK documentation, then calculate "
        "the sum of 15 and 27, then format the result as a percentage summary."
    )
    
    # Build learning_state with playbook entries matching the runtime flow
    # This matches how learning_state is structured when loaded from database
    test_learning_state = {
        "metadata": {
            "playbook_entries": PLAYBOOK_ENTRIES.copy(),
        },
    }
    
    # Run the task with test_learning_state parameter
    # This ensures playbook entries are available when resolve_playbook() is called
    # during Student/Teacher initialization, matching the actual runtime flow
    # Agentic adapters work in all execution modes (auto, paired, coach)
    try:
        result = await atlas_arun(
            task=task,
            config_path=config_path,
            session_metadata={
                "source": "tool_adoption_validation",
                # Removed execution_mode - agentic adapters work in all modes
            },
            stream_progress=False,
            test_learning_state=test_learning_state,
        )
    except Exception as exc:
        return False, f"Task execution failed: {exc}"
    
    # Re-get context after arun completes (it may have been reset)
    context = ExecutionContext.get()
    
    # Validate adoption tracking
    metadata = context.metadata
    learning_usage = metadata.get("learning_usage", {})
    
    issues = []
    
    # Check that learning_usage exists
    if not learning_usage:
        issues.append("learning_usage metadata missing")
        return False, "; ".join(issues) if issues else "Missing learning_usage"
    
    # Check session-level metrics
    session_block = learning_usage.get("session", {})
    cue_hits = session_block.get("cue_hits", 0)
    action_adoptions = session_block.get("action_adoptions", 0)
    
    if cue_hits == 0:
        issues.append("No cue hits detected (expected > 0)")
    
    if action_adoptions == 0:
        issues.append("No action adoptions detected (expected > 0)")
    
    # Check role-level adoption records
    roles = learning_usage.get("roles", {})
    student_roles = roles.get("student", {})
    
    if not student_roles:
        issues.append("No student role entries found")
    else:
        # Verify at least one entry has adoption records
        found_adoption = False
        for entry_id, entry_data in student_roles.items():
            if isinstance(entry_data, dict):
                adoptions = entry_data.get("action_adoptions", 0)
                runtime_handle = entry_data.get("runtime_handle", "")
                if adoptions > 0:
                    found_adoption = True
                    # Verify runtime_handle matches expected tool names
                    if runtime_handle not in ["web_search", "calculate", "format_text"]:
                        issues.append(
                            f"Unexpected runtime_handle in adoption: {runtime_handle}"
                        )
        
        if not found_adoption:
            issues.append("No adoption records found in student role entries")
    
    # Check that adoption_steps are recorded
    unique_adoption_steps = session_block.get("unique_adoption_steps", [])
    if not unique_adoption_steps:
        issues.append("No adoption step IDs recorded")
    
    # Verify runtime_handles are populated from tool calls
    runtime_handles = metadata.get("runtime_handles", [])
    if not runtime_handles:
        issues.append("Runtime handles not populated (expected from tool calls)")
    else:
        # Verify at least one expected handle is present
        expected_handles = {"web_search", "calculate", "format_text"}
        found_expected = any(handle in expected_handles for handle in runtime_handles)
        if not found_expected:
            issues.append(
                f"Expected runtime handles not found. Got: {runtime_handles}"
            )
    
    if issues:
        return False, "; ".join(issues)
    
    # Success summary
    summary = (
        f"✓ Cue hits: {cue_hits}, "
        f"✓ Action adoptions: {action_adoptions}, "
        f"✓ Adoption steps: {len(unique_adoption_steps)}, "
        f"✓ Student entries: {len(student_roles)}, "
        f"✓ Runtime handles: {runtime_handles}"
    )
    
    return True, summary


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate tool-backed adoption tracking"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file with tool-enabled adapter",
    )
    
    args = parser.parse_args()
    
    success, message = asyncio.run(validate_tool_adoption(args.config))
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

