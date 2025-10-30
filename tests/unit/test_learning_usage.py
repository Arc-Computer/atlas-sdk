from __future__ import annotations

from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext


def test_learning_usage_tracker_records_hits_and_adoptions():
    context = ExecutionContext.get()
    context.reset()
    context.metadata["learning_usage_config"] = {"enabled": True, "capture_examples": True, "max_examples_per_entry": 1}
    tracker = get_tracker(context)
    tracker.register_entries(
        "student",
        [
            {
                "id": "entry-1",
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
    tracker.record_action_adoption(
        "student",
        "logs.search",
        success=False,
        step_id=8,
        metadata={"status": "failed"},
    )
    tracker.record_session_outcome(
        reward_score=0.75,
        token_usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200, "calls": 3},
        incident_id="INC-123",
        task_identifier="Checkout outage",
        incident_tags=["sre", "customer-impact", "sre"],
        retry_count=2,
        failure_flag=True,
        failure_events=[{"step_id": 3, "status": "failed"}],
    )
    usage = context.metadata.get("learning_usage")
    assert usage is not None
    assert usage["roles"]["student"]["entry-1"]["cue_hits"] == 1
    assert usage["roles"]["student"]["entry-1"]["action_adoptions"] == 2
    assert usage["roles"]["student"]["entry-1"]["successful_adoptions"] == 1
    assert usage["roles"]["student"]["entry-1"]["failed_adoptions"] == 1
    assert usage["session"]["cue_hits"] == 1
    assert usage["session"]["action_adoptions"] == 2
    assert usage["session"]["failed_adoptions"] == 1
    assert usage["session"]["reward_score"] == 0.75
    assert usage["session"]["token_usage"]["total_tokens"] == 200
    assert usage["session"]["incident_id"] == "INC-123"
    assert usage["session"]["task_identifier"] == "Checkout outage"
    assert usage["session"]["retry_count"] == 2
    assert usage["session"]["failure_flag"] is True
    assert usage["session"]["failure_events"] == [{"step_id": 3, "status": "failed"}]
