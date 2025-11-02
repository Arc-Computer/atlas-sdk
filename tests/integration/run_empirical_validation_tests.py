#!/usr/bin/env python3
"""Standalone test runner for empirical validation tests.

This bypasses pytest plugin loading issues and runs tests directly.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

# Add project to path
sys.path.insert(0, '.')

from atlas.config.models import LearningConfig, PlaybookPruningConfig
from atlas.learning.synthesizer import LearningSynthesizer


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
    print("✓ test_provisional_entry_has_validation_metadata")


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
    print("✓ test_validated_entry_has_active_lifecycle")


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
    print("✓ test_impact_initialization")


def test_prune_too_specific_entry():
    """Test pruning entry that is too specific (low cue hit rate)."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_prune_too_specific_entry")


def test_prune_harmful_entry():
    """Test pruning entry that is harmful (negative reward delta)."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_prune_harmful_entry")


def test_prune_neutral_entry():
    """Test pruning entry that is neutral (no improvement despite adoption)."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_prune_neutral_entry")


def test_prune_no_transfer_entry():
    """Test pruning entry with no transfer success."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_prune_no_transfer_entry")


def test_keep_good_entry():
    """Test that good entries are kept."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_keep_good_entry")


def test_keep_entry_with_insufficient_data():
    """Test that entries with insufficient data are kept."""
    pruning_config = PlaybookPruningConfig()
    learning_config = LearningConfig(enabled=False, provisional_acceptance=True, pruning_config=pruning_config)
    synthesizer = LearningSynthesizer(config=learning_config, client=None)
    
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
    print("✓ test_keep_entry_with_insufficient_data")


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
    print("✓ test_transfer_success_detection")


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
    print("✓ test_metrics_computation")


def main():
    """Run all tests."""
    print("Running empirical validation tests...\n")
    
    tests = [
        test_provisional_entry_has_validation_metadata,
        test_validated_entry_has_active_lifecycle,
        test_impact_initialization,
        test_prune_too_specific_entry,
        test_prune_harmful_entry,
        test_prune_neutral_entry,
        test_prune_no_transfer_entry,
        test_keep_good_entry,
        test_keep_entry_with_insufficient_data,
        test_transfer_success_detection,
        test_metrics_computation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

