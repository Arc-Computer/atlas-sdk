"""Atlas SDK helper surface for self-managed agent discovery."""

from __future__ import annotations

from atlas.sdk.decorators import agent, environment
from atlas.sdk.interfaces import (
    AtlasAgentProtocol,
    AtlasEnvironmentProtocol,
    DiscoveryContext,
    DiscoveryEvent,
    TelemetryEmitterProtocol,
)

__all__ = [
    "agent",
    "environment",
    "AtlasAgentProtocol",
    "AtlasEnvironmentProtocol",
    "DiscoveryContext",
    "DiscoveryEvent",
    "TelemetryEmitterProtocol",
]
