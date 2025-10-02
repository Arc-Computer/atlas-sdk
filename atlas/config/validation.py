# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit CLI validation utilities."""

from __future__ import annotations

from pathlib import Path


from atlas.config.loader import load_config
from atlas.config.models import AtlasConfig

def validate_config(config_path: Path) -> AtlasConfig:
    return load_config(config_path)
