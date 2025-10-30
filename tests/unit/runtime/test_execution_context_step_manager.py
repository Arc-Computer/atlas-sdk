import logging

import pytest

from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.telemetry.langchain_callback import TelemetryCallbackHandler


def test_intermediate_step_manager_cached_across_accesses():
    context = ExecutionContext.get()
    context.reset()
    manager_one = context.intermediate_step_manager
    manager_two = ExecutionContext.get().intermediate_step_manager
    assert manager_one is manager_two

    context.reset()
    manager_three = context.intermediate_step_manager
    assert manager_three is not manager_one


@pytest.mark.asyncio
async def test_telemetry_callbacks_reuse_step_manager(caplog):
    ExecutionContext.get().reset()
    handler = TelemetryCallbackHandler()

    caplog.set_level(logging.WARNING, logger="atlas.runtime.orchestration.step_manager")
    await handler.on_tool_start({"name": "tool"}, "{}", "run-123")
    await handler.on_tool_end("{}", "run-123")

    warning_text = "Step id run-123 not found in outstanding start steps"
    assert warning_text not in caplog.text
