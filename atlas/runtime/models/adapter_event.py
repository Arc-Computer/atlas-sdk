# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-originated telemetry events streamed during execution."""

from __future__ import annotations

import time
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


AdapterEventName = Literal["env_action", "tool_response", "progress", "error"]


class AdapterTelemetryEvent(BaseModel):
    """Structured telemetry emitted by self-managed adapters."""

    model_config = ConfigDict(extra="allow")

    event: AdapterEventName
    payload: Any | None = None
    reason: str | None = None
    step: int | None = None
    timestamp: float = Field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] | None = None

    def envelope(self) -> Dict[str, Any]:
        """Return a serialisable dict for downstream consumers."""

        payload = {
            "event": self.event,
            "payload": self.payload,
            "reason": self.reason,
            "step": self.step,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return {key: value for key, value in payload.items() if value is not None}


__all__ = ["AdapterEventName", "AdapterTelemetryEvent"]
