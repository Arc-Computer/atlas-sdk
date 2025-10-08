import asyncio
import os

import pytest

from langchain_core.messages import AIMessage

from atlas.connectors.registry import build_adapter
from atlas.config.models import (
    AdapterType,
    LLMParameters,
    LLMProvider,
    OpenAIAdapterConfig,
    StudentConfig,
    StudentPrompts,
)
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.personas.student import Student
from atlas.prompts import build_student_prompts
from atlas.types import Step


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
        reasoning_effort="medium",
    )


def _student_config() -> StudentConfig:
    return StudentConfig(
        prompts=StudentPrompts(
            planner=(
                "{base_prompt}\nRespond with JSON containing a single key 'steps'."
                " Each step entry must include the keys 'id', 'description', 'depends_on', 'tool', and 'tool_params'."
                " Step IDs must be integers starting from 1."
            ),
            executor="{base_prompt}\nReturn the execution result as JSON with content and trace fields.",
            synthesizer="{base_prompt}\nProduce a concise final answer.",
        )
    )


def test_student_plan_execute_and_synthesize_live():
    if os.getenv("OPENAI_API_KEY") is None:
        pytest.skip("requires OPENAI_API_KEY for live adapter test")
    async def runner() -> None:
        ExecutionContext.get().reset()
        adapter_config = OpenAIAdapterConfig(
            type=AdapterType.OPENAI,
            name="student-openai",
            system_prompt="You are the Atlas student agent generating plans, executing steps, and summarizing results.",
            tools=[],
            llm=_gpt5_params(),
        )
        adapter = build_adapter(adapter_config)
        last_response: dict[str, object] = {}
        original_adapter_call = adapter.ainvoke

        async def traced_adapter(prompt, metadata=None):
            try:
                result = await original_adapter_call(prompt, metadata)
            except Exception:
                raw = last_response.get("content", "")
                if raw:
                    print(f"Student adapter raw response: {raw}")
                raise
            last_response["content"] = result
            return result

        adapter.ainvoke = traced_adapter
        student_cfg = _student_config()
        student_prompts = build_student_prompts(adapter_config.system_prompt, student_cfg)
        student = Student(
            adapter=adapter,
            adapter_config=adapter_config,
            student_config=student_cfg,
            student_prompts=student_prompts,
        )
        task = "Draft a short Atlas progress update referencing benchmarks."
        try:
            plan = await student.acreate_plan(task)
        except Exception:
            raw = last_response.get("content", "")
            if raw:
                print(f"Student planning raw response: {raw}")
            else:
                print(f"Student planning payload: {last_response.get('raw', {})}")
            raise
        assert plan.steps
        step = plan.steps[0]
        try:
            execution = await student.aexecute_step(step, context={}, guidance=[])
        except Exception:
            raw = last_response.get("content", "")
            if raw:
                print(f"Student execution raw response: {raw}")
            else:
                print(f"Student execution payload: {last_response.get('raw', {})}")
            raise
        assert execution.trace.strip()
        assert execution.output.strip()
        try:
            final_answer = await student.asynthesize_final_answer(task, [])
        except Exception:
            raw = last_response.get("content", "")
            if raw:
                print(f"Student synthesis raw response: {raw}")
            else:
                print(f"Student synthesis payload: {last_response.get('raw', {})}")
            raise
        assert final_answer.strip()

    asyncio.run(runner())


def test_student_extracts_reasoning_metadata():
    student = object.__new__(Student)
    message = AIMessage(
        content="result",
        additional_kwargs={
            "reasoning_content": [{"type": "thought", "text": "consider options"}],
            "thinking_blocks": [{"type": "analysis", "content": "details"}],
        },
    )
    metadata = student._extract_reasoning_metadata([message])
    assert "reasoning" in metadata
    reasoning_entry = metadata["reasoning"][0]
    assert reasoning_entry["payload"]["reasoning_content"][0]["text"] == "consider options"
    trace = student._build_trace([message])
    assert "AI_REASONING" in trace


def test_student_handles_langgraph_stream_events():
    ExecutionContext.get().reset()
    ExecutionContext.get().metadata["active_actor"] = "student"
    student = object.__new__(Student)
    student._llm_stream_state = {}
    step = Step(id=1, description="demo", tool=None, tool_params=None)
    captured = []
    manager = ExecutionContext.get().intermediate_step_manager
    subscription = manager.subscribe(lambda event: captured.append(event))
    try:
        student._handle_stream_event(
            step,
            {
                "event": "on_chain_start",
                "name": "agent",
                "metadata": {"langgraph_node": "agent"},
                "run_id": "task-run",
                "data": {"input": {"state": "init"}},
            },
            manager,
        )
        assert captured[-1].payload.event_type == IntermediateStepType.TASK_START

        student._handle_stream_event(
            step,
            {
                "event": "on_chat_model_start",
                "name": "ChatOpenAI",
                "metadata": {"langgraph_node": "agent"},
                "run_id": "llm-run",
                "data": {"input": {"messages": []}},
            },
            manager,
        )
        assert captured[-1].payload.event_type == IntermediateStepType.LLM_START

        student._handle_stream_event(
            step,
            {
                "event": "on_chat_model_stream",
                "name": "ChatOpenAI",
                "metadata": {"langgraph_node": "agent"},
                "run_id": "llm-run",
                "data": {"chunk": {"content": "Hello"}},
            },
            manager,
        )
        stream_event = captured[-1].payload
        assert stream_event.event_type == IntermediateStepType.LLM_NEW_TOKEN
        assert stream_event.data.chunk["text"] == "Hello"
        assert stream_event.data.chunk["token_counts"]["accumulated"] >= 1

        student._handle_stream_event(
            step,
            {
                "event": "on_chat_model_end",
                "name": "ChatOpenAI",
                "metadata": {"langgraph_node": "agent"},
                "run_id": "llm-run",
                "data": {"output": {"content": "Done"}},
            },
            manager,
        )
        final_event = captured[-1].payload
        assert final_event.event_type == IntermediateStepType.LLM_END
        assert final_event.metadata["token_counts"]["approx_total"] >= 1
    finally:
        subscription.unsubscribe()
