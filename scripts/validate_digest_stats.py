#!/usr/bin/env python3
"""Validation script for digest_stats telemetry capture.

This script validates that prompt digest_stats are captured end-to-end in .atlas/runs artifacts:
1. Runs a task with Claude/Gemini config (must use live provider access)
2. Inspects .atlas/runs/run_{timestamp}.json files
3. Verifies digest_stats are present with all required fields

Usage:
    python scripts/validate_digest_stats.py --config configs/eval/learning/tool_adoption_claude.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.core import arun as atlas_arun
from atlas.utils.env import load_dotenv_if_available


def find_latest_run_artifact(atlas_dir: Path) -> Path | None:
    """Find the most recent run artifact file."""
    runs_dir = atlas_dir / ".atlas" / "runs"
    if not runs_dir.exists():
        return None
    
    run_files = sorted(runs_dir.glob("run_*.json"), reverse=True)
    return run_files[0] if run_files else None


def validate_digest_stats_in_artifact(artifact_path: Path) -> tuple[bool, str]:
    """Validate that digest_stats are present in the artifact."""
    try:
        with artifact_path.open() as f:
            artifact_data = json.load(f)
    except Exception as exc:
        return False, f"Failed to read artifact: {exc}"
    
    # Check for digest_stats in metadata
    metadata = artifact_data.get("metadata", {})
    digest_stats = metadata.get("digest_stats")
    
    if not digest_stats:
        # Try root level as fallback
        digest_stats = artifact_data.get("digest_stats")
    
    if not digest_stats:
        return False, "digest_stats not found in metadata or root level"
    
    if not isinstance(digest_stats, dict):
        return False, f"digest_stats is not a dict: {type(digest_stats)}"
    
    # Verify required fields
    required_fields = ["size", "budget", "util"]
    missing_fields = [field for field in required_fields if field not in digest_stats]
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    issues = []
    
    # Validate field types
    size = digest_stats.get("size")
    budget = digest_stats.get("budget")
    util = digest_stats.get("util")
    
    if not isinstance(size, int) or size < 0:
        issues.append(f"size must be non-negative int, got {size} ({type(size)})")
    
    if not isinstance(budget, int) or budget <= 0:
        issues.append(f"budget must be positive int, got {budget} ({type(budget)})")
    
    if not isinstance(util, (int, float)) or not (0 <= util <= 1):
        issues.append(f"util must be float 0-1, got {util} ({type(util)})")
    
    # Check optional fields (should exist but not required)
    sections = digest_stats.get("sections")
    if sections is not None and not isinstance(sections, dict):
        issues.append(f"sections must be dict, got {type(sections)}")
    
    # Validate provider-specific budget (Claude ~20k, Gemini ~100k)
    # But allow some flexibility
    if budget < 10000:
        issues.append(f"budget seems too low for provider: {budget}")
    elif budget > 200000:
        issues.append(f"budget seems too high: {budget}")
    
    if issues:
        return False, "; ".join(issues)
    
    # Success summary
    summary = (
        f"✓ size: {size}, "
        f"✓ budget: {budget}, "
        f"✓ util: {util:.4f}, "
        f"✓ sections: {len(sections) if isinstance(sections, dict) else 0}"
    )
    
    if digest_stats.get("omitted"):
        omitted_count = len(digest_stats["omitted"])
        summary += f", omitted keys: {omitted_count}"
    
    if digest_stats.get("omitted_sections"):
        omitted_sections = digest_stats["omitted_sections"]
        summary += f", omitted sections: {omitted_sections}"
    
    return True, summary


async def validate_digest_stats(config_path: str) -> tuple[bool, str]:
    """Run a task and validate digest_stats capture."""
    load_dotenv_if_available()
    
    # Simple task that will trigger digest creation
    task = "Analyze the key features of the Atlas SDK and provide a summary."
    
    # Run the task
    try:
        result = await atlas_arun(
            task=task,
            config_path=config_path,
            session_metadata={"source": "digest_stats_validation"},
            stream_progress=False,
        )
    except Exception as exc:
        return False, f"Task execution failed: {exc}"
    
    # Find the latest run artifact
    atlas_dir = Path.cwd()
    artifact_path = find_latest_run_artifact(atlas_dir)
    
    if not artifact_path:
        return False, "No run artifact found in .atlas/runs/"
    
    # Validate digest_stats in artifact
    return validate_digest_stats_in_artifact(artifact_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate digest_stats telemetry capture"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (must use Claude/Gemini provider with live access)",
    )
    
    args = parser.parse_args()
    
    success, message = asyncio.run(validate_digest_stats(args.config))
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")
    
    if success:
        print(f"\nArtifact location: .atlas/runs/run_*.json")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

