#!/usr/bin/env python3
"""Validation script for empirical learning metrics.

This script loads learning state from the database and validates that:
1. Provisional entries are accepted and tracked
2. Impact metrics are computed correctly
3. Pruning logic works as expected
4. Transfer success is detected correctly

Usage:
    python scripts/validate_empirical_learning.py <learning_key> [--config <config_path>]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from atlas.config.loader import load_config
from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database


def _safe_ratio(numerator: float | None, denominator: int | None) -> float | None:
    """Safely compute ratio, returning None if denominator is zero."""
    if denominator is None or denominator == 0:
        return None
    if numerator is None:
        return None
    return numerator / denominator


def compute_entry_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    """Compute all validation metrics for an entry."""
    impact = entry.get("impact", {})
    if not isinstance(impact, dict):
        return {}
    
    sessions_observed = impact.get("sessions_observed", 0)
    sessions_with_hits = impact.get("sessions_with_hits", 0)
    total_cue_hits = impact.get("total_cue_hits", 0)
    successful_adoptions = impact.get("successful_adoptions", 0)
    
    # Cue hit rate
    cue_hit_rate = (
        sessions_with_hits / sessions_observed
        if sessions_observed > 0
        else 0.0
    )
    
    # Adoption rate
    adoption_rate = (
        successful_adoptions / total_cue_hits
        if total_cue_hits > 0
        else 0.0
    )
    
    # Reward delta
    reward_with = _safe_ratio(
        impact.get("reward_with_sum"),
        impact.get("reward_with_count"),
    )
    reward_without = _safe_ratio(
        impact.get("reward_without_sum"),
        impact.get("reward_without_count"),
    )
    reward_delta = (
        (reward_with - reward_without)
        if (reward_with is not None and reward_without is not None)
        else None
    )
    
    # Token delta
    tokens_with = _safe_ratio(
        impact.get("tokens_with_sum"),
        impact.get("tokens_with_count"),
    )
    tokens_without = _safe_ratio(
        impact.get("tokens_without_sum"),
        impact.get("tokens_without_count"),
    )
    token_delta = (
        (tokens_with - tokens_without)
        if (tokens_with is not None and tokens_without is not None)
        else None
    )
    
    # Transfer success
    incident_ids = impact.get("incident_ids", [])
    if not isinstance(incident_ids, list):
        incident_ids = []
    transfer_success = len(incident_ids) >= 2
    
    return {
        "cue_hit_rate": cue_hit_rate,
        "adoption_rate": adoption_rate,
        "reward_delta": reward_delta,
        "token_delta": token_delta,
        "transfer_success": transfer_success,
        "unique_incidents": len(set(incident_ids)),
        "sessions_observed": sessions_observed,
        "sessions_with_hits": sessions_with_hits,
        "total_cue_hits": total_cue_hits,
        "successful_adoptions": successful_adoptions,
    }


def check_pruning_criteria(
    metrics: dict[str, Any],
    pruning_config: dict[str, Any],
) -> list[str]:
    """Check if entry should be pruned based on criteria."""
    reasons = []
    
    min_sessions = pruning_config.get("min_sessions", 10)
    min_cue_hit_rate = pruning_config.get("min_cue_hit_rate", 0.05)
    min_reward_delta = pruning_config.get("min_reward_delta", 0.01)
    min_transfer_sessions = pruning_config.get("min_transfer_sessions", 20)
    
    sessions_observed = metrics["sessions_observed"]
    sessions_with_hits = metrics["sessions_with_hits"]
    cue_hit_rate = metrics["cue_hit_rate"]
    reward_delta = metrics["reward_delta"]
    adoption_rate = metrics["adoption_rate"]
    transfer_success = metrics["transfer_success"]
    
    # Too specific
    if cue_hit_rate < min_cue_hit_rate and sessions_observed >= min_sessions:
        reasons.append("too_specific")
    
    # Harmful
    if (
        reward_delta is not None
        and reward_delta < -min_reward_delta
        and sessions_with_hits >= 5
    ):
        reasons.append("harmful")
    
    # Neutral
    if (
        reward_delta is not None
        and reward_delta < min_reward_delta
        and adoption_rate > 0.5
        and sessions_with_hits >= 10
    ):
        reasons.append("neutral")
    
    # No transfer
    if not transfer_success and sessions_observed >= min_transfer_sessions:
        reasons.append("no_transfer")
    
    return reasons


def format_entry_summary(entry: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Format entry summary for display."""
    entry_id = entry.get("id", "unknown")
    audience = entry.get("audience", "unknown")
    cue_pattern = entry.get("cue", {}).get("pattern", "N/A") if isinstance(entry.get("cue"), dict) else "N/A"
    
    provenance = entry.get("provenance", {})
    status = provenance.get("status", {}) if isinstance(provenance, dict) else {}
    lifecycle = status.get("lifecycle", "unknown") if isinstance(status, dict) else "unknown"
    
    validation_status = entry.get("metadata", {}).get("validation_status", "unknown")
    validation_warnings = entry.get("metadata", {}).get("validation_warnings", [])
    
    lines = [
        f"Entry: {entry_id}",
        f"  Audience: {audience}",
        f"  Cue Pattern: {cue_pattern}",
        f"  Lifecycle: {lifecycle}",
        f"  Validation Status: {validation_status}",
    ]
    
    if validation_warnings:
        lines.append(f"  Validation Warnings: {len(validation_warnings)}")
        for warning in validation_warnings[:3]:  # Show first 3
            lines.append(f"    - {warning}")
    
    lines.extend([
        f"  Sessions Observed: {metrics['sessions_observed']}",
        f"  Sessions with Hits: {metrics['sessions_with_hits']}",
        f"  Cue Hit Rate: {metrics['cue_hit_rate']:.3f}",
        f"  Adoption Rate: {metrics['adoption_rate']:.3f}",
        f"  Reward Delta: {metrics['reward_delta']:.3f}" if metrics['reward_delta'] is not None else "  Reward Delta: N/A",
        f"  Transfer Success: {metrics['transfer_success']}",
        f"  Unique Incidents: {metrics['unique_incidents']}",
    ])
    
    return "\n".join(lines)


async def validate_learning_state(
    learning_key: str,
    config_path: str | None,
) -> int:
    """Validate learning state for a given learning key."""
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        # Try to find default config
        import os
        default_paths = [
            "atlas-postgres.yaml",
            "configs/examples/openai_agent.yaml",
        ]
        config = None
        for path in default_paths:
            if os.path.exists(path):
                config = load_config(path)
                break
        
        if config is None:
            print("ERROR: No config file found. Please specify --config")
            return 1
    
    # Connect to database
    if not config.storage:
        print("ERROR: No storage configuration found in config")
        return 1
    
    database = Database(config.storage)
    try:
        await database.connect()
        
        # Fetch learning state
        learning_state = await database.fetch_learning_state(learning_key)
        if not learning_state:
            print(f"ERROR: No learning state found for key: {learning_key}")
            return 1
        
        metadata = learning_state.get("metadata", {})
        if not isinstance(metadata, dict):
            print("ERROR: Invalid metadata structure")
            return 1
        
        entries = metadata.get("playbook_entries", [])
        if not isinstance(entries, list):
            print("ERROR: playbook_entries is not a list")
            return 1
        
        print(f"Found {len(entries)} playbook entries\n")
        
        # Get pruning config from metadata or use defaults
        pruning_config = metadata.get("pruning_config", {})
        if not isinstance(pruning_config, dict):
            pruning_config = {
                "min_sessions": 10,
                "min_cue_hit_rate": 0.05,
                "min_reward_delta": 0.01,
                "min_transfer_sessions": 20,
            }
        
        # Validate each entry
        provisional_count = 0
        validated_count = 0
        pruned_count = 0
        active_count = 0
        should_prune_count = 0
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            
            metrics = compute_entry_metrics(entry)
            
            # Check lifecycle
            provenance = entry.get("provenance", {})
            status = provenance.get("status", {}) if isinstance(provenance, dict) else {}
            lifecycle = status.get("lifecycle", "unknown") if isinstance(status, dict) else "unknown"
            
            if lifecycle == "pruned":
                pruned_count += 1
            elif lifecycle == "active":
                active_count += 1
            
            # Check validation status
            validation_status = entry.get("metadata", {}).get("validation_status")
            if validation_status == "provisional":
                provisional_count += 1
            elif validation_status == "validated":
                validated_count += 1
            
            # Check if should be pruned
            prune_reasons = check_pruning_criteria(metrics, pruning_config)
            if prune_reasons and lifecycle != "pruned":
                should_prune_count += 1
                print(f"⚠️  Entry should be pruned (reasons: {', '.join(prune_reasons)}):")
                print(format_entry_summary(entry, metrics))
                print()
            elif lifecycle == "pruned":
                prune_reason = provenance.get("prune_reason", "unknown")
                print(f"✓ Pruned entry (reason: {prune_reason}):")
                print(format_entry_summary(entry, metrics))
                print()
            elif lifecycle == "active":
                print(f"✓ Active entry:")
                print(format_entry_summary(entry, metrics))
                print()
        
        # Summary
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Entries: {len(entries)}")
        print(f"  Provisional: {provisional_count}")
        print(f"  Validated: {validated_count}")
        print(f"  Active: {active_count}")
        print(f"  Pruned: {pruned_count}")
        print(f"  Should Prune (but not pruned): {should_prune_count}")
        
        if should_prune_count > 0:
            print("\n⚠️  WARNING: Some entries should be pruned but aren't!")
            return 1
        
        print("\n✓ Validation passed!")
        return 0
        
    finally:
        await database.disconnect()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate empirical learning metrics",
    )
    parser.add_argument(
        "learning_key",
        help="Learning key to validate",
    )
    parser.add_argument(
        "--config",
        help="Path to config file (optional)",
    )
    
    args = parser.parse_args()
    
    return asyncio.run(validate_learning_state(args.learning_key, args.config))


if __name__ == "__main__":
    sys.exit(main())

