from __future__ import annotations

from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext


def test_learning_usage_tracker_records_hits_and_adoptions():
    context = ExecutionContext.get()
    context.reset()
    context.metadata["learning_usage_config"] = {"enabled": True, "capture_examples": True, "max_examples_per_nugget": 1}
    tracker = get_tracker(context)
    tracker.register_nuggets(
        "student",
        [
            {
                "id": "nugget-1",
                "cue": {"type": "keyword", "pattern": "check logs"},
                "action": {"imperative": "Call log search", "runtime_handle": "logs.search"},
                "scope": {"category": "reinforcement"},
            }
        ],
    )
    tracker.detect_and_record("student", "Always check logs before querying", step_id=7, context_hint="step context")
    tracker.record_action_adoption(
        "student",
        "logs.search",
        success=True,
        step_id=7,
        metadata={"status": "ok"},
    )
    usage = context.metadata.get("learning_usage")
    assert usage is not None
    assert usage["roles"]["student"]["nugget-1"]["cue_hits"] == 1
    assert usage["roles"]["student"]["nugget-1"]["action_adoptions"] == 1
    assert usage["roles"]["student"]["nugget-1"]["successful_adoptions"] == 1
    assert usage["session"]["cue_hits"] == 1
    assert usage["session"]["action_adoptions"] == 1
