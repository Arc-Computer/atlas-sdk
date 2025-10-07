import asyncio

import pytest

from langchain_core.messages import AIMessage

from atlas.agent.registry import build_adapter
from atlas.config.models import (
    AdapterType,
    LLMParameters,
    LLMProvider,
    OpenAIAdapterConfig,
    PromptRewriteConfig,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.orchestration.execution_context import ExecutionContext
from atlas.roles.student import Student
from atlas.transition.rewriter import PromptRewriteEngine


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
    )


def _wrap_rewrite_client(engine: PromptRewriteEngine) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = engine._client.acomplete

    async def traced(messages, response_format=None, overrides=None):
        merged = dict(overrides or {})
        extra_body = dict(merged.get("extra_body") or {})
        extra_body.setdefault("reasoning_effort", "medium")
        merged["extra_body"] = extra_body
        response = await original(messages, response_format, merged)
        captured["content"] = response.content
        captured["raw"] = response.raw
        return response

    engine._client.acomplete = traced
    return captured


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
        rewrite_engine = PromptRewriteEngine(
            PromptRewriteConfig(llm=_gpt5_params(), max_tokens=4096, temperature=1.0),
            fallback_llm=None,
        )
        rewrite_capture = _wrap_rewrite_client(rewrite_engine)
        student_cfg = _student_config()
        teacher_cfg = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=3072,
            plan_cache_seconds=0,
            guidance_max_tokens=1536,
            validation_max_tokens=1536,
        )
        try:
            student_prompts, _ = await rewrite_engine.generate(
                base_prompt=adapter_config.system_prompt,
                adapter_config=adapter_config,
                student_config=student_cfg,
                teacher_config=teacher_cfg,
            )
        except Exception:
            raw = rewrite_capture.get("content", "")
            if raw:
                print(f"Prompt rewrite raw response: {raw}")
            else:
                print(f"Prompt rewrite payload: {rewrite_capture.get('raw', {})}")
            raise
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
