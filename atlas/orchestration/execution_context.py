# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.builder.context."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from atlas.data_models.intermediate_step import IntermediateStep
from atlas.data_models.intermediate_step import IntermediateStepPayload
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.data_models.intermediate_step import StreamEventData
from atlas.data_models.invocation_node import InvocationNode
from atlas.utils.reactive.subject import Subject

if typing.TYPE_CHECKING:
    from atlas.types import StepEvaluation


class _Singleton(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ExecutionContextState(metaclass=_Singleton):
    def __init__(self) -> None:
        self._event_stream: ContextVar[Subject[IntermediateStep] | None] = ContextVar("event_stream", default=None)
        self._active_function: ContextVar[InvocationNode | None] = ContextVar("active_function", default=None)
        self._active_span_id_stack: ContextVar[list[str] | None] = ContextVar("active_span_id_stack", default=None)
        self._metadata: ContextVar[dict[str, typing.Any] | None] = ContextVar("execution_metadata", default=None)

    @property
    def event_stream(self) -> ContextVar[Subject[IntermediateStep]]:
        if self._event_stream.get() is None:
            self._event_stream.set(Subject())
        return typing.cast(ContextVar[Subject[IntermediateStep]], self._event_stream)

    @property
    def active_function(self) -> ContextVar[InvocationNode]:
        if self._active_function.get() is None:
            self._active_function.set(InvocationNode(function_id="root", function_name="root"))
        return typing.cast(ContextVar[InvocationNode], self._active_function)

    @property
    def active_span_id_stack(self) -> ContextVar[list[str]]:
        if self._active_span_id_stack.get() is None:
            self._active_span_id_stack.set(["root"])
        return typing.cast(ContextVar[list[str]], self._active_span_id_stack)

    @property
    def metadata(self) -> ContextVar[dict[str, typing.Any]]:
        if self._metadata.get() is None:
            self._metadata.set({})
        return typing.cast(ContextVar[dict[str, typing.Any]], self._metadata)

    @staticmethod
    def get() -> ExecutionContextState:
        return ExecutionContextState()


class ActiveFunctionHandle:
    def __init__(self) -> None:
        self._output: typing.Any | None = None

    @property
    def output(self) -> typing.Any | None:
        return self._output

    def set_output(self, output: typing.Any) -> None:
        self._output = output


class ExecutionContext:
    def __init__(self, state: ExecutionContextState) -> None:
        self._state = state

    @property
    def metadata(self) -> dict[str, typing.Any]:
        return self._state.metadata.get()

    @property
    def active_function(self) -> InvocationNode:
        return self._state.active_function.get()

    @property
    def active_span_id(self) -> str:
        return self._state.active_span_id_stack.get()[-1]

    @property
    def event_stream(self) -> Subject[IntermediateStep]:
        return self._state.event_stream.get()

    def reset(self) -> None:
        self._state.metadata.set({})
        self._state.event_stream.set(Subject())
        self._state.active_function.set(InvocationNode(function_id="root", function_name="root"))
        self._state.active_span_id_stack.set(["root"])

    def _step_metadata(self, step_id: int) -> dict:
        metadata = self._state.metadata.get()
        steps = metadata.setdefault("steps", {})
        return steps.setdefault(step_id, {"attempts": [], "guidance": []})

    def register_step_attempt(self, step_id: int, attempt: int, evaluation: "StepEvaluation | dict") -> None:
        entry = self._step_metadata(step_id)
        if hasattr(evaluation, "to_dict"):
            payload = typing.cast("StepEvaluation", evaluation).to_dict()
        else:
            payload = typing.cast(dict, evaluation)
        entry.setdefault("attempts", []).append({"attempt": attempt, "evaluation": payload})

    def append_guidance(self, step_id: int, guidance: str) -> None:
        entry = self._step_metadata(step_id)
        entry.setdefault("guidance", []).append(guidance)

    @property
    def intermediate_step_manager(self) -> "IntermediateStepManager":
        from atlas.orchestration.step_manager import IntermediateStepManager

        return IntermediateStepManager(self._state)

    @contextmanager
    def push_active_function(
        self,
        function_name: str,
        input_data: typing.Any | None,
        metadata: dict[str, typing.Any] | None = None,
    ) -> Iterator[ActiveFunctionHandle]:
        parent = self._state.active_function.get()
        function_id = str(uuid.uuid4())
        node = InvocationNode(
            function_id=function_id,
            function_name=function_name,
            parent_id=parent.function_id,
            parent_name=parent.function_name,
        )
        token = self._state.active_function.set(node)
        handle = ActiveFunctionHandle()
        step_manager = self.intermediate_step_manager
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=function_id,
                event_type=IntermediateStepType.FUNCTION_START,
                name=function_name,
                data=StreamEventData(input=input_data),
                metadata=metadata,
            )
        )
        try:
            yield handle
        finally:
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=function_id,
                    event_type=IntermediateStepType.FUNCTION_END,
                    name=function_name,
                    data=StreamEventData(input=input_data, output=handle.output),
                )
            )
            self._state.active_function.reset(token)

    @staticmethod
    def get() -> ExecutionContext:
        return ExecutionContext(ExecutionContextState.get())


__all__ = ["ExecutionContext", "ExecutionContextState", "ActiveFunctionHandle"]
