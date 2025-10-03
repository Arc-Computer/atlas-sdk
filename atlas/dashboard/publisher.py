"""Telemetry event publisher for the Atlas dashboard."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from typing import TYPE_CHECKING

from atlas.data_models.intermediate_step import IntermediateStep
from atlas.utils.reactive.subscription import Subscription

if TYPE_CHECKING:
    from atlas.orchestration.step_manager import IntermediateStepManager


class TelemetryEventBus:
    """In-process fan-out queue for telemetry events."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop

    def publish_payload(self, payload: dict[str, Any]) -> None:
        loop = self._ensure_loop()
        if loop.is_closed():
            return

        def dispatch() -> None:
            for queue in list(self._queues):
                queue.put_nowait(payload)

        running = self._current_loop()
        if running is loop:
            dispatch()
        else:
            loop.call_soon_threadsafe(dispatch)

    async def iterate(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues.add(queue)
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            self._queues.discard(queue)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self.bind_loop()
        assert self._loop is not None
        return self._loop

    @staticmethod
    def _current_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None


class TelemetryPublisher:
    """Connects the orchestration event stream to the dashboard."""

    def __init__(self, event_bus: TelemetryEventBus | None = None) -> None:
        self._event_bus = event_bus or TelemetryEventBus()
        self._subscription: Subscription[IntermediateStep] | None = None

    @property
    def event_bus(self) -> TelemetryEventBus:
        return self._event_bus

    def attach(self, step_manager: "IntermediateStepManager") -> None:
        self._event_bus.bind_loop()
        self.detach()
        self._subscription = step_manager.subscribe(self._handle_intermediate_step)

    def detach(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    def publish_control_event(self, event_type: str, data: dict[str, Any]) -> None:
        self._event_bus.publish_payload({"type": event_type, "data": data})

    def stream(self) -> AsyncIterator[dict[str, Any]]:
        return self._event_bus.iterate()

    def _handle_intermediate_step(self, event: IntermediateStep) -> None:
        payload = {
            "type": "intermediate-step",
            "data": event.model_dump(mode="json"),
        }
        self._event_bus.publish_payload(payload)
