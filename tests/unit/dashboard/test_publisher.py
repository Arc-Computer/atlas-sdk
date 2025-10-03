import asyncio

from atlas.dashboard.publisher import TelemetryPublisher
from atlas.data_models.intermediate_step import IntermediateStep
from atlas.data_models.intermediate_step import IntermediateStepPayload
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.data_models.intermediate_step import StreamEventData
from atlas.data_models.invocation_node import InvocationNode


class FakeSubscription:
    def __init__(self, unsubscribe):
        self._unsubscribe = unsubscribe

    def unsubscribe(self):
        self._unsubscribe()


class FakeManager:
    def __init__(self):
        self._on_next = None

    def subscribe(self, on_next, *_):
        self._on_next = on_next
        return FakeSubscription(lambda: None)

    def emit(self, event: IntermediateStep):
        assert self._on_next is not None
        self._on_next(event)


def build_event(name: str) -> IntermediateStep:
    payload = IntermediateStepPayload(
        event_type=IntermediateStepType.FUNCTION_START,
        name=name,
        data=StreamEventData(input={"value": name}),
    )
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(function_id="root", function_name="root"),
        payload=payload,
    )


def test_publisher_forwards_events():
    async def runner():
        publisher = TelemetryPublisher()
        manager = FakeManager()
        publisher.attach(manager)
        stream = publisher.stream()

        first_task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        manager.emit(build_event("alpha"))
        first = await first_task
        assert first["type"] == "intermediate-step"
        assert first["data"]["payload"]["name"] == "alpha"

        second_task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        publisher.publish_control_event("session-started", {"session_id": 1})
        second = await second_task
        assert second["type"] == "session-started"
        assert second["data"]["session_id"] == 1

        await stream.aclose()
        publisher.detach()

    asyncio.run(runner())
