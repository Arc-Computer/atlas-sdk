"""Script to validate learning configuration across all config files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.config.loader import load_config
from atlas.config.models import LearningConfig


def validate_learning_config(config_path: str) -> tuple[bool, str]:
    """Validate learning config in a config file."""
    try:
        config = load_config(config_path)
        learning = config.learning
        
        # Check required fields for empirical validation
        issues = []
        
        if not learning.enabled:
            issues.append("Learning is disabled")
        
        # Note: provisional_acceptance defaults to True, enforce_generality defaults to False
        # So we don't need to check them - they're defaults
        
        # Check pruning config exists (has defaults, but good to verify)
        if not learning.pruning_config:
            issues.append("pruning_config is missing")
        else:
            pc = learning.pruning_config
            if pc.min_sessions < 1 or pc.min_sessions > 100:
                issues.append(f"pruning_config.min_sessions should be 1-100, got {pc.min_sessions}")
            if pc.min_cue_hit_rate < 0 or pc.min_cue_hit_rate > 1:
                issues.append(f"pruning_config.min_cue_hit_rate should be 0-1, got {pc.min_cue_hit_rate}")
            if pc.min_transfer_sessions < 1 or pc.min_transfer_sessions > 100:
                issues.append(f"pruning_config.min_transfer_sessions should be 1-100, got {pc.min_transfer_sessions}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "OK"
    except Exception as e:
        return False, f"Error loading config: {e}"


def main():
    """Validate all config files."""
    configs = [
        "configs/examples/openai_agent.yaml",
        "configs/examples/python_agent.yaml",
        "configs/examples/http_agent.yaml",
        "atlas/templates/openai_agent.yaml",
        "docker/configs/atlas.docker.yaml",
        "examples/mcp_tool_learning/config.yaml",
    ]
    
    all_valid = True
    for config_path in configs:
        path = Path(config_path)
        if not path.exists():
            print(f"⚠️  {config_path}: File not found")
            continue
        
        valid, message = validate_learning_config(config_path)
        status = "✅" if valid else "❌"
        print(f"{status} {config_path}: {message}")
        if not valid:
            all_valid = False
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

