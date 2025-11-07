"""Tests for structured task/step payload passing to BYOA adapters."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from atlas.connectors.python import PythonAdapter
from atlas.connectors.langchain_bridge import BYOABridgeLLM
from atlas.config.models import PythonAdapterConfig
from atlas.runtime.orchestration.execution_context import ExecutionContext


@pytest.fixture
def mock_python_adapter_callable():
    """Mock callable that captures metadata for inspection."""
    captured = {"calls": []}

    def callable_fn(prompt: str, metadata: dict = None):
        captured["calls"].append({"prompt": prompt, "metadata": metadata})
        # Return simple JSON plan for planning, simple output for execution
        if metadata and metadata.get("mode") == "planning":
            return json.dumps({"steps": [{"id": 1, "description": "test", "depends_on": []}]})
        return json.dumps({"result": "executed"})

    callable_fn.captured = captured
    return callable_fn


@pytest.fixture
def python_adapter_config():
    """Minimal Python adapter config."""
    return PythonAdapterConfig(
        type="python",
        name="test",
        import_path="test_module",
        attribute="test_fn",
        system_prompt="",  # Required field
    )


class TestStructuredPayloadPassing:
    """Test that structured payloads flow through to adapters."""

    @pytest.mark.asyncio
    async def test_planning_phase_receives_task_payload(self, python_adapter_config, mock_python_adapter_callable):
        """Verify task_payload is passed during planning phase."""
        with patch.object(PythonAdapter, "_load_callable", return_value=mock_python_adapter_callable):
            adapter = PythonAdapter(python_adapter_config)

            # Simulate planning phase call
            task = "test task for planning"
            metadata = {
                "mode": "planning",
                "task_payload": task,
            }

            result = await adapter.ainvoke("Generate a plan for: test task", metadata=metadata)

            # Verify adapter received the call
            assert len(mock_python_adapter_callable.captured["calls"]) == 1
            call = mock_python_adapter_callable.captured["calls"][0]

            # Verify structured payload was passed
            assert call["metadata"]["mode"] == "planning"
            assert call["metadata"]["task_payload"] == task

    @pytest.mark.asyncio
    async def test_execution_phase_receives_step_payload(self, python_adapter_config, mock_python_adapter_callable):
        """Verify step_payload is passed during execution phase."""
        with patch.object(PythonAdapter, "_load_callable", return_value=mock_python_adapter_callable):
            adapter = PythonAdapter(python_adapter_config)

            # Simulate execution phase call
            task = "test task"
            step_payload = {
                "step_id": 1,
                "description": "Execute test step",
                "depends_on": [],
            }
            metadata = {
                "task_payload": task,
                "step_payload": step_payload,
            }

            result = await adapter.ainvoke("Execute step 1", metadata=metadata)

            # Verify adapter received the call
            assert len(mock_python_adapter_callable.captured["calls"]) == 1
            call = mock_python_adapter_callable.captured["calls"][0]

            # Verify structured payloads were passed
            assert call["metadata"]["task_payload"] == task
            assert call["metadata"]["step_payload"] == step_payload
            assert call["metadata"]["step_payload"]["step_id"] == 1

    @pytest.mark.asyncio
    async def test_langchain_bridge_enriches_metadata_from_context(self):
        """Verify LangChain bridge adds structured payloads from ExecutionContext when adapter supports it."""
        from langchain_core.messages import HumanMessage

        # Create mock adapter that captures metadata AND supports structured payloads
        mock_adapter = AsyncMock()
        mock_adapter.ainvoke = AsyncMock(return_value="test response")
        mock_adapter.supports_structured_payloads = True  # Enable enrichment

        # Create bridge
        bridge = BYOABridgeLLM(mock_adapter, tool_definitions=[])

        # Setup execution context with task
        context = ExecutionContext.get()
        context.metadata["task"] = "test task from context"

        # Call bridge with step_payload in config (simulating LangGraph execution)
        messages = [HumanMessage(content="test")]

        # Mock run_manager with metadata containing step_payload
        mock_run_manager = MagicMock()
        mock_run_manager.metadata = {
            "step_payload": {
                "step_id": 2,
                "description": "Test step from context",
                "depends_on": [1],
            }
        }

        result = await bridge._agenerate(messages, run_manager=mock_run_manager)

        # Verify adapter was called with enriched metadata
        assert mock_adapter.ainvoke.called
        call_args = mock_adapter.ainvoke.call_args

        # Extract metadata argument
        metadata = call_args[1]["metadata"]

        # Verify structured payloads were added
        assert metadata["task_payload"] == "test task from context"
        assert metadata["step_payload"]["step_id"] == 2
        assert metadata["step_payload"]["description"] == "Test step from context"

    @pytest.mark.asyncio
    async def test_langchain_bridge_gracefully_handles_missing_context(self):
        """Verify bridge works when ExecutionContext is not available."""
        from langchain_core.messages import HumanMessage

        # Create mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.ainvoke = AsyncMock(return_value="test response")

        # Create bridge
        bridge = BYOABridgeLLM(mock_adapter, tool_definitions=[])

        # Clear execution context metadata
        try:
            context = ExecutionContext.get()
            context.metadata.clear()
        except Exception:
            pass

        # Call bridge
        messages = [HumanMessage(content="test")]
        result = await bridge._agenerate(messages)

        # Verify adapter was still called (graceful degradation)
        assert mock_adapter.ainvoke.called

        # Metadata may be empty or minimal (no structured payloads)
        call_args = mock_adapter.ainvoke.call_args
        metadata = call_args[1].get("metadata")

        # Should not have structured payloads, but shouldn't crash
        if metadata:
            assert "task_payload" not in metadata or metadata.get("task_payload") is None
            assert "step_payload" not in metadata or metadata.get("step_payload") is None

    def test_structured_payload_size_is_reasonable(self):
        """Verify structured payloads are lightweight (~1-2KB)."""
        # Create realistic payloads
        task = "Create a new customer record with name 'John Doe' and email 'john@example.com'"
        step_payload = {
            "step_id": 1,
            "description": "Call create_customer API with provided details",
            "depends_on": [],
        }

        # Serialize to JSON (what gets stored/transmitted)
        metadata = {
            "mode": "execution",
            "task_payload": task,
            "step_payload": step_payload,
        }

        serialized = json.dumps(metadata)
        size_bytes = len(serialized.encode("utf-8"))

        # Verify size is reasonable (should be < 1KB for typical payloads)
        assert size_bytes < 1024, f"Payload too large: {size_bytes} bytes"
        print(f"Payload size: {size_bytes} bytes (target: < 1KB)")


class TestStudentIntegration:
    """Integration tests for Student persona with structured payloads."""

    @pytest.mark.asyncio
    async def test_student_acreate_plan_passes_task_payload(self):
        """Verify Student.acreate_plan passes task_payload during planning."""
        from unittest.mock import patch, AsyncMock
        from atlas.personas.student import Student
        from atlas.config.models import StudentConfig, AdapterConfig
        from atlas.prompts import RewrittenStudentPrompts

        # Track adapter calls
        adapter_calls = []

        async def mock_ainvoke(prompt: str, metadata: dict = None):
            adapter_calls.append({"prompt": prompt, "metadata": metadata})
            return json.dumps({"steps": [{"id": 1, "description": "test", "depends_on": []}]})

        # Create mock adapter with supports_structured_payloads
        mock_adapter = AsyncMock()
        mock_adapter.ainvoke = mock_ainvoke
        mock_adapter.supports_structured_payloads = True

        # Create Student with mock adapter
        student_config = StudentConfig()
        adapter_config = AdapterConfig(
            type="python",
            name="test",
            tools=[],
            system_prompt="",
        )
        prompts = RewrittenStudentPrompts(
            planner="test planner",
            executor="test executor",
            synthesizer="test synthesizer",
        )

        with patch('atlas.connectors.langchain_bridge.build_bridge', return_value=(AsyncMock(), [])):
            student = Student(mock_adapter, adapter_config, student_config, prompts)

            # Call acreate_plan
            task = "test task for planning"
            plan = await student.acreate_plan(task)

            # Verify adapter was called with task_payload
            assert len(adapter_calls) == 1
            call = adapter_calls[0]
            assert call["metadata"] is not None
            assert call["metadata"].get("mode") == "planning"
            assert call["metadata"].get("task_payload") == task

    @pytest.mark.asyncio
    async def test_student_aexecute_step_concurrent_payloads(self):
        """Verify concurrent steps receive correct step_payloads (no race condition)."""
        from unittest.mock import patch, AsyncMock, MagicMock
        from atlas.personas.student import Student
        from atlas.types import Step
        from atlas.config.models import StudentConfig, AdapterConfig
        from atlas.prompts import RewrittenStudentPrompts
        import asyncio

        # Track step payloads seen by each execution
        step_payloads_seen = {}

        async def mock_ainvoke(prompt: str, metadata: dict = None):
            metadata = metadata or {}
            step_payload = metadata.get("step_payload")
            if step_payload:
                step_id = step_payload.get("step_id")
                step_payloads_seen[step_id] = step_payload
            return json.dumps({"result": "executed"})

        mock_adapter = AsyncMock()
        mock_adapter.ainvoke = mock_ainvoke
        mock_adapter.supports_structured_payloads = True

        student_config = StudentConfig()
        adapter_config = AdapterConfig(type="python", name="test", tools=[], system_prompt="")
        prompts = RewrittenStudentPrompts(
            planner="test", executor="test", synthesizer="test"
        )

        # Mock the graph to avoid complex LangGraph setup
        mock_graph = AsyncMock()

        async def mock_astream_events(state, config=None, version=None):
            # Extract step_payload from config and simulate minimal event stream
            step_payload = config.get("metadata", {}).get("step_payload") if config else None
            if step_payload:
                # Simulate calling the adapter via bridge
                await mock_adapter.ainvoke("test prompt", metadata={"step_payload": step_payload})
            # Yield minimal events
            yield {"event": "on_chat_model_end", "data": {"chunk": {"content": "done"}}}

        mock_graph.astream_events = mock_astream_events

        with patch('atlas.connectors.langchain_bridge.build_bridge', return_value=(AsyncMock(), [])):
            student = Student(mock_adapter, adapter_config, student_config, prompts)
            student._graph = mock_graph

            # Execute two steps concurrently
            step1 = Step(id=1, description="Step 1", depends_on=[])
            step2 = Step(id=2, description="Step 2", depends_on=[])

            results = await asyncio.gather(
                student.aexecute_step(step1, {}, None),
                student.aexecute_step(step2, {}, None),
            )

            # Verify each step saw its own payload (no cross-contamination)
            assert 1 in step_payloads_seen
            assert 2 in step_payloads_seen
            assert step_payloads_seen[1]["step_id"] == 1
            assert step_payloads_seen[1]["description"] == "Step 1"
            assert step_payloads_seen[2]["step_id"] == 2
            assert step_payloads_seen[2]["description"] == "Step 2"


class TestBackwardCompatibility:
    """Verify existing adapters continue to work without structured payloads."""

    @pytest.mark.asyncio
    async def test_adapter_without_structured_payload_support(self, python_adapter_config):
        """Adapters that don't use structured payloads should work unchanged."""

        # Legacy adapter that ignores metadata
        def legacy_callable(prompt: str, metadata: dict = None):
            return "response from prompt parsing only"

        with patch.object(PythonAdapter, "_load_callable", return_value=legacy_callable):
            adapter = PythonAdapter(python_adapter_config)

            # Call with structured payloads
            metadata = {
                "mode": "planning",
                "task_payload": "some task",
                "step_payload": {"step_id": 1, "description": "step"},
            }

            result = await adapter.ainvoke("test prompt", metadata=metadata)

            # Should still work - adapter ignores structured payloads
            assert result == "response from prompt parsing only"

    @pytest.mark.asyncio
    async def test_adapter_called_without_metadata(self, python_adapter_config):
        """Adapters should handle being called without any metadata."""

        def simple_callable(prompt: str, metadata: dict = None):
            return f"processed: {prompt}"

        with patch.object(PythonAdapter, "_load_callable", return_value=simple_callable):
            adapter = PythonAdapter(python_adapter_config)

            # Call without metadata
            result = await adapter.ainvoke("test prompt", metadata=None)

            assert result == "processed: test prompt"
