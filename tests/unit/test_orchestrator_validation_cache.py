import asyncio
import json

from atlas.config.models import AdaptiveTeachingConfig, LLMParameters, LLMProvider, OrchestrationConfig, RIMConfig
from atlas.personas.student import StudentStepResult
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.types import Step


class _CachingTeacherStub:
    def __init__(self) -> None:
        self.calls = 0
        self.signature_value = "sig-1"

    def validation_signature(self, step, structured_output, prior_results, prior_guidance, attempt_guidance):
        return self.signature_value

    async def avalidate_step(self, step, trace, structured_output, prior_results, prior_guidance, attempt_guidance):
        self.calls += 1
        return {
            "valid": False,
            "guidance": "retry",
            "cached": False,
            "validation_request": {"hash": "abc", "preview": "payload"},
        }


class _RepeatingStudentStub:
    def __init__(self, structured_output):
        self._structured_output = structured_output
        self.calls = 0

    async def aexecute_step(self, step, context_outputs, guidance):
        self.calls += 1
        payload = json.dumps(self._structured_output)
        return StudentStepResult(
            trace=f"trace-{self.calls}",
            output=payload,
            messages=[],
            metadata={},
            artifacts={},
            deliverable=None,
        )


class _EvaluatorStub:
    pass


def _llm_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=0.0,
        timeout_seconds=30.0,
    )


async def _run_cached_flow():
    ExecutionContext.get().reset()
    teacher = _CachingTeacherStub()
    student = _RepeatingStudentStub(
        {
            "status": "ok",
            "result": {"deliverable": "value"},
            "text": "value",
        }
    )
    evaluator = _EvaluatorStub()
    orchestrator = Orchestrator(
        teacher=teacher,
        student=student,
        evaluator=evaluator,
        orchestration_config=OrchestrationConfig(max_retries=1),
        rim_config=RIMConfig(small_model=_llm_params(), large_model=_llm_params()),
        adaptive_config=AdaptiveTeachingConfig(),
    )
    step = Step(id=1, description="demo", depends_on=[])
    execution_context = ExecutionContext.get()
    await orchestrator._run_step(
        task="demo",
        step=step,
        context_outputs={},
        execution_context=execution_context,
        require_validation=True,
        allow_retry=True,
    )
    await orchestrator._run_step(
        task="demo",
        step=step,
        context_outputs={},
        execution_context=execution_context,
        require_validation=True,
        allow_retry=False,
    )
    return teacher, dict(ExecutionContext.get().metadata)


def test_orchestrator_reuses_validation_cache():
    teacher, metadata = asyncio.run(_run_cached_flow())
    assert teacher.calls == 1
    cache = metadata.get("validation_cache")
    assert cache and teacher.signature_value in cache
    attempts = metadata.get("steps", {}).get(1, {}).get("attempts", [])
    assert len(attempts) == 3
    assert attempts[-1]["evaluation"]["validation"].get("cached") is True
    assert "validation_request" in attempts[-1]["evaluation"]["validation"]
