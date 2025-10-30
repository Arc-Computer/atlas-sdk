from __future__ import annotations

from atlas.cli.run import _render_learning_summary


def test_render_learning_summary_with_rich_metadata() -> None:
    metadata = {
        "reward_summary": {
            "score": 0.86,
            "recent_mean": 0.82,
            "baseline_mean": 0.75,
            "recent_count": 5,
            "baseline_count": 20,
        },
        "session_reward_stats": {
            "score": 0.86,
            "recent_mean": 0.82,
            "baseline_mean": 0.75,
            "recent_count": 5,
            "baseline_count": 20,
        },
        "token_usage": {
            "prompt_tokens": 123,
            "completion_tokens": 321,
            "total_tokens": 444,
            "calls": 3,
        },
        "learning_usage": {
            "session": {
                "cue_hits": 2,
                "action_adoptions": 3,
                "failed_adoptions": 1,
                "unique_cue_steps": [1, 2],
                "unique_adoption_steps": [3],
            },
            "roles": {
                "student": {
                    "entry-alpha": {
                        "cue_hits": 2,
                        "action_adoptions": 2,
                        "successful_adoptions": 1,
                        "failed_adoptions": 1,
                        "runtime_handle": "validate_assumptions",
                    }
                }
            },
        },
        "session_student_learning": "Focus on summarising outcomes.",
        "session_teacher_learning": "Coach them to cite metrics.",
        "session_learning_note": "Combine telemetry insights.",
        "teacher_notes": ["Encourage playbook use.", "Escalate when uncertain."],
        "learning_state": {
            "metadata": {
                "playbook_entries": [
                    {
                        "id": "entry-alpha",
                        "action": {"runtime_handle": "validate_assumptions"},
                        "provenance": {"status": {"lifecycle": "active"}},
                        "impact": {
                            "total_cue_hits": 4,
                            "total_adoptions": 3,
                            "successful_adoptions": 2,
                            "failed_adoptions": 1,
                            "reward_with_sum": 5.1,
                            "reward_with_count": 3,
                            "reward_without_sum": 2.4,
                            "reward_without_count": 2,
                            "tokens_with_sum": 600,
                            "tokens_with_count": 3,
                            "tokens_without_sum": 500,
                            "tokens_without_count": 2,
                        },
                    },
                    {
                        "id": "entry-beta",
                        "provenance": {"status": {"lifecycle": "deprecated"}},
                    },
                ],
                "last_failure": {
                    "timestamp": "2025-10-29T12:00:00Z",
                    "rejected_candidates": [
                        {
                            "id": "entry-gamma",
                            "status": {"category": "differentiation", "lifecycle": "rejected"},
                        }
                    ],
                },
            }
        },
        "learning_key": "tenant::agent",
    }

    summary = _render_learning_summary(metadata)

    assert "=== Learning Summary ===" in summary
    assert "Reward: latest=0.86; recent=0.82 (n=5); baseline=0.75 (n=20); Δ=+0.07" in summary
    assert "Tokens: prompt=123 completion=321 total=444 calls=3" in summary
    assert "Usage: cue_hits=2 (steps=2); adoptions=3 (success=2, failed=1) steps=1" in summary
    assert "Learning Notes:" in summary
    assert "Student: Focus on summarising outcomes." in summary
    assert "Teacher: Coach them to cite metrics." in summary
    assert "Active Playbook Entries:" in summary
    assert "entry-alpha [validate_assumptions]" in summary
    assert "Recent Failures: rejected=1 (latest: 2025-10-29T12:00:00Z)" in summary
    assert "Learning Key: tenant::agent" in summary


def test_render_learning_summary_handles_missing_sections() -> None:
    summary = _render_learning_summary({})
    assert summary == ""


def test_render_learning_summary_stream_mode_header() -> None:
    metadata = {"reward_summary": {"score": 0.5}}
    summary = _render_learning_summary(metadata, stream=True)
    assert summary.startswith("--- Learning Summary ---")
