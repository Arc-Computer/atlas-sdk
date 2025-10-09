# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.builder.intermediate_step_manager."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from atlas.runtime.models import IntermediateStep
from atlas.runtime.models import IntermediateStepPayload
from atlas.runtime.models import IntermediateStepState
from atlas.runtime.orchestration.execution_context import ExecutionContextState
from atlas.utils.reactive.observable import OnComplete
from atlas.utils.reactive.observable import OnError
from atlas.utils.reactive.observable import OnNext
from atlas.utils.reactive.subscription import Subscription

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _OpenStep:
    step_id: str
    step_name: str
    step_type: str
    step_parent_id: str
    prev_stack: list[str]
    active_stack: list[str]


class IntermediateStepManager:
    def __init__(self, state: ExecutionContextState) -> None:
        self._state = state
        self._open_steps: dict[str, _OpenStep] = {}

    def push_intermediate_step(self, payload: IntermediateStepPayload) -> None:
        if not isinstance(payload, IntermediateStepPayload):
            raise TypeError("payload must be an IntermediateStepPayload instance")
        span_stack = self._state.active_span_id_stack.get()
        if payload.event_state == IntermediateStepState.START:
            parent_step_id = span_stack[-1]
            next_stack = span_stack + [payload.UUID]
            self._state.active_span_id_stack.set(next_stack)
            self._open_steps[payload.UUID] = _OpenStep(
                step_id=payload.UUID,
                step_name=payload.name or payload.UUID,
                step_type=payload.event_type,
                step_parent_id=parent_step_id,
                prev_stack=span_stack,
                active_stack=next_stack,
            )
            logger.debug(
                "Pushed start step %s name=%s type=%s parent=%s",
                payload.UUID,
                payload.name,
                payload.event_type,
                parent_step_id,
            )
        elif payload.event_state == IntermediateStepState.END:
            open_step = self._open_steps.pop(payload.UUID, None)
            if open_step is None:
                logger.warning("Step id %s not found in outstanding start steps", payload.UUID)
                return
            parent_step_id = open_step.step_parent_id
            current_stack = open_step.active_stack
            prev_stack = open_step.prev_stack
            self._state.active_span_id_stack.set(prev_stack)
            while current_stack and current_stack[-1] != parent_step_id:
                current_stack.pop()
            if current_stack != prev_stack:
                logger.warning("Active span stack mismatch when closing step %s", payload.UUID)
            logger.debug(
                "Popped end step %s name=%s type=%s parent=%s",
                payload.UUID,
                payload.name,
                payload.event_type,
                parent_step_id,
            )
        elif payload.event_state == IntermediateStepState.CHUNK:
            open_step = self._open_steps.get(payload.UUID)
            if open_step is None:
                logger.warning("Chunk received without matching start for step %s", payload.UUID)
                return
            parent_step_id = open_step.step_parent_id
        else:
            raise ValueError(f"Unsupported event state {payload.event_state}")
        active_function = self._state.active_function.get()
        intermediate_step = IntermediateStep(
            parent_id=parent_step_id,
            function_ancestry=active_function,
            payload=payload,
        )
        self._state.event_stream.get().on_next(intermediate_step)

    def subscribe(
        self,
        on_next: OnNext[IntermediateStep],
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> Subscription:
        return self._state.event_stream.get().subscribe(on_next, on_error, on_complete)

__all__ = ["IntermediateStepManager"]
