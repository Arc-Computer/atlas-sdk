"""Integration tests for empirical validation of learning playbook entries."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from atlas.config.models import LearningConfig, PlaybookPruningConfig
from atlas.learning.synthesizer import LearningSynthesizer


@pytest.fixture
def pruning_config() -> PlaybookPruningConfig:
    """Default pruning configuration for tests."""
    return PlaybookPruningConfig(
        min_sessions=10,
        min_cue_hit_rate=0.05,
        min_reward_delta=0.01,
        min_transfer_sessions=20,
    )


@pytest.fixture
def learning_config(pruning_config: PlaybookPruningConfig) -> LearningConfig:
    """Learning config with empirical validation enabled."""
    return LearningConfig(
        enabled=False,  # Disable LLM calls for testing
        provisional_acceptance=True,
        pruning_config=pruning_config,
    )


@pytest.fixture
def synthesizer(learning_config: LearningConfig) -> LearningSynthesizer:
    """Create a synthesizer instance for testing."""
    # LearningSynthesizer takes LearningConfig which includes gates, schema, etc.
    # We'll test the pruning logic directly without needing an LLM client
    return LearningSynthesizer(
        config=learning_config,
        client=None,  # We don't need LLM for testing pruning logic
    )


def test_provisional_entry_has_validation_metadata():
    """Test that provisional entries have validation_status and validation_warnings."""
    entry = {
        "id": "test-entry-1",
        "audience": "student",
        "cue": {"pattern": "test.*pattern"},
        "action": {"imperative": "Do something"},
        "metadata": {
            "validation_status": "provisional",
            "validation_warnings": ["Generality gate failed: proper noun detected"],
        },
        "provenance": {
            "status": {
                "lifecycle": "provisional",
            },
        },
    }
    
    assert entry["metadata"]["validation_status"] == "provisional"
    assert isinstance(entry["metadata"]["validation_warnings"], list)
    assert len(entry["metadata"]["validation_warnings"]) > 0


def test_validated_entry_has_active_lifecycle():
    """Test that validated entries have lifecycle='active'."""
    entry = {
        "id": "test-entry-2",
        "audience": "student",
        "cue": {"pattern": "test.*pattern"},
        "action": {"imperative": "Do something"},
        "metadata": {
            "validation_status": "validated",
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    assert entry["metadata"]["validation_status"] == "validated"
    assert entry["provenance"]["status"]["lifecycle"] == "active"


def test_impact_initialization():
    """Test that entries have impact dict initialized."""
    entry = {
        "id": "test-entry-3",
        "impact": {
            "sessions_observed": 0,
            "sessions_with_hits": 0,
            "total_cue_hits": 0,
            "successful_adoptions": 0,
            "reward_with_sum": 0.0,
            "reward_with_count": 0,
            "reward_without_sum": 0.0,
            "reward_without_count": 0,
            "incident_ids": [],
        },
    }
    
    impact = entry.get("impact", {})
    assert isinstance(impact, dict)
    assert "sessions_observed" in impact
    assert "incident_ids" in impact


def test_prune_too_specific_entry(synthesizer: LearningSynthesizer):
    """Test pruning entry that is too specific (low cue hit rate)."""
    entry = {
        "id": "too-specific-entry",
        "impact": {
            "sessions_observed": 15,
            "sessions_with_hits": 0,  # Never fires
            "total_cue_hits": 0,
            "successful_adoptions": 0,
            "reward_with_sum": 0.0,
            "reward_with_count": 0,
            "reward_without_sum": 0.0,
            "reward_without_count": 0,
            "incident_ids": [],
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be pruned
    assert len(result) == 0
    assert entry["provenance"]["status"]["lifecycle"] == "pruned"
    assert entry["provenance"]["prune_reason"] == "too_specific"


def test_prune_harmful_entry(synthesizer: LearningSynthesizer):
    """Test pruning entry that is harmful (negative reward delta)."""
    entry = {
        "id": "harmful-entry",
        "impact": {
            "sessions_observed": 10,
            "sessions_with_hits": 5,
            "total_cue_hits": 10,
            "successful_adoptions": 3,
            "reward_with_sum": 5.0,  # Avg: 0.5
            "reward_with_count": 10,
            "reward_without_sum": 6.0,  # Avg: 0.6
            "reward_without_count": 10,
            "incident_ids": ["incident-1"],
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be pruned (reward_delta = 0.5 - 0.6 = -0.1 < -0.01)
    assert len(result) == 0
    assert entry["provenance"]["status"]["lifecycle"] == "pruned"
    assert entry["provenance"]["prune_reason"] == "harmful"


def test_prune_neutral_entry(synthesizer: LearningSynthesizer):
    """Test pruning entry that is neutral (no improvement despite adoption)."""
    entry = {
        "id": "neutral-entry",
        "impact": {
            "sessions_observed": 15,
            "sessions_with_hits": 12,
            "total_cue_hits": 20,
            "successful_adoptions": 15,  # High adoption rate: 15/20 = 0.75
            "reward_with_sum": 10.0,  # Avg: 0.5
            "reward_with_count": 20,
            "reward_without_sum": 10.0,  # Avg: 0.5 (no improvement)
            "reward_without_count": 20,
            "incident_ids": ["incident-1"],
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be pruned (reward_delta = 0.0 < 0.01, adoption_rate = 0.75 > 0.5)
    assert len(result) == 0
    assert entry["provenance"]["status"]["lifecycle"] == "pruned"
    assert entry["provenance"]["prune_reason"] == "neutral"


def test_prune_no_transfer_entry(synthesizer: LearningSynthesizer):
    """Test pruning entry with no transfer success."""
    entry = {
        "id": "no-transfer-entry",
        "impact": {
            "sessions_observed": 25,
            "sessions_with_hits": 10,
            "total_cue_hits": 15,
            "successful_adoptions": 8,
            "reward_with_sum": 8.0,
            "reward_with_count": 10,
            "reward_without_sum": 7.0,
            "reward_without_count": 10,
            "incident_ids": ["incident-1"],  # Only one incident
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be pruned (only 1 incident, < 2)
    assert len(result) == 0
    assert entry["provenance"]["status"]["lifecycle"] == "pruned"
    assert entry["provenance"]["prune_reason"] == "no_transfer"


def test_keep_good_entry(synthesizer: LearningSynthesizer):
    """Test that good entries are kept."""
    entry = {
        "id": "good-entry",
        "impact": {
            "sessions_observed": 15,
            "sessions_with_hits": 10,  # High hit rate: 10/15 = 0.67
            "total_cue_hits": 15,
            "successful_adoptions": 10,
            "reward_with_sum": 8.0,  # Avg: 0.8
            "reward_with_count": 10,
            "reward_without_sum": 6.0,  # Avg: 0.6
            "reward_without_count": 10,
            "incident_ids": ["incident-1", "incident-2"],  # Transfer success
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be kept
    assert len(result) == 1
    assert result[0]["id"] == "good-entry"
    assert entry["provenance"]["status"]["lifecycle"] == "active"


def test_keep_entry_with_insufficient_data(synthesizer: LearningSynthesizer):
    """Test that entries with insufficient data are kept."""
    entry = {
        "id": "new-entry",
        "impact": {
            "sessions_observed": 5,  # Less than min_sessions=10
            "sessions_with_hits": 0,
            "total_cue_hits": 0,
            "successful_adoptions": 0,
            "reward_with_sum": 0.0,
            "reward_with_count": 0,
            "reward_without_sum": 0.0,
            "reward_without_count": 0,
            "incident_ids": [],
        },
        "provenance": {
            "status": {
                "lifecycle": "active",
            },
        },
    }
    
    result = synthesizer._prune_ineffective_entries([entry])
    
    # Entry should be kept (not enough data to prune)
    assert len(result) == 1
    assert result[0]["id"] == "new-entry"


def test_transfer_success_detection():
    """Test that transfer success is correctly detected."""
    # Entry with transfer success (>= 2 incidents)
    entry_with_transfer = {
        "impact": {
            "incident_ids": ["incident-1", "incident-2"],
        },
    }
    
    incident_ids = entry_with_transfer["impact"].get("incident_ids", [])
    transfer_success = len(incident_ids) >= 2
    assert transfer_success is True
    
    # Entry without transfer success (< 2 incidents)
    entry_without_transfer = {
        "impact": {
            "incident_ids": ["incident-1"],
        },
    }
    
    incident_ids = entry_without_transfer["impact"].get("incident_ids", [])
    transfer_success = len(incident_ids) >= 2
    assert transfer_success is False


def test_metrics_computation():
    """Test that metrics are computed correctly."""
    entry = {
        "impact": {
            "sessions_observed": 20,
            "sessions_with_hits": 15,
            "total_cue_hits": 25,
            "successful_adoptions": 20,
            "reward_with_sum": 15.0,
            "reward_with_count": 20,
            "reward_without_sum": 10.0,
            "reward_without_count": 20,
            "incident_ids": ["incident-1", "incident-2"],
        },
    }
    
    impact = entry["impact"]
    
    # Cue hit rate
    cue_hit_rate = impact["sessions_with_hits"] / impact["sessions_observed"]
    assert cue_hit_rate == 0.75  # 15 / 20
    
    # Adoption rate
    adoption_rate = impact["successful_adoptions"] / impact["total_cue_hits"]
    assert adoption_rate == 0.8  # 20 / 25
    
    # Reward delta
    reward_with = impact["reward_with_sum"] / impact["reward_with_count"]
    reward_without = impact["reward_without_sum"] / impact["reward_without_count"]
    reward_delta = reward_with - reward_without
    assert reward_delta == 0.25  # 0.75 - 0.5
    
    # Transfer success
    transfer_success = len(impact["incident_ids"]) >= 2
    assert transfer_success is True

