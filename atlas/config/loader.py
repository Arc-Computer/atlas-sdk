# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loader adapted from NeMo Agent Toolkit runtime loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from atlas.config.models import AtlasConfig

class ConfigLoadError(RuntimeError):
    """Raised when a configuration file cannot be loaded or validated."""

def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigLoadError(f"configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"failed to read configuration: {exc}") from exc
    return data

def load_config(config_path: str | Path) -> AtlasConfig:
    path = Path(config_path).expanduser().resolve()
    payload = _read_yaml(path)
    return parse_config(payload)

def parse_config(data: dict[str, Any]) -> AtlasConfig:
    try:
        return AtlasConfig.model_validate(data)
    except Exception as exc:
        raise ConfigLoadError(f"configuration validation failed: {exc}") from exc

__all__ = ["ConfigLoadError", "load_config", "parse_config"]
