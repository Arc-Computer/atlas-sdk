"""Factory helpers wrapping the adapter registry."""

from __future__ import annotations

from atlas.agent.registry import AgentAdapter
from atlas.agent.registry import build_adapter
from atlas.config.models import AdapterUnion
from atlas.config.models import AtlasConfig


def create_adapter(config: AdapterUnion) -> AgentAdapter:
    return build_adapter(config)


def create_from_atlas_config(config: AtlasConfig) -> AgentAdapter:
    return build_adapter(config.agent)


__all__ = ["create_adapter", "create_from_atlas_config"]
