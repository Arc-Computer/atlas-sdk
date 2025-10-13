"""Integration tests for the persona memory learning engine."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas import core
from atlas.config.models import (
    AdapterConfig,
    AdapterType,
    LLMParameters,
    OrchestrationConfig,
    RIMConfig,
    StorageConfig,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.prompts import RewrittenStudentPrompts, RewrittenTeacherPrompts
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.persona_memory import build_fingerprint, extract_fingerprint_inputs, get_cache
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Result, StepEvaluation, StepResult

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


def _apply_schema_statements() -> List[str]:
    schema_path = Path(__file__).resolve().parents[2] / "atlas" / "runtime" / "storage" / "schema.sql"
    return [
        statement.strip()
        for statement in schema_path.read_text().split(";")
        if statement.strip()
    ]


async def _apply_schema(database: Database) -> None:
    statements = _apply_schema_statements()
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        for statement in statements:
            await connection.execute(statement)


class StubAdapter:
    async def ainvoke(self, prompt: str, metadata=None):
        return "{}"


class StubStudent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update_prompts(self, student_prompts: RewrittenStudentPrompts) -> None:
        pass


class StubTeacher:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update_prompts(self, prompts: RewrittenTeacherPrompts) -> None:
        pass


class StubEvaluator:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FailingThenRecoveringOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        self._persona_refresh = kwargs.get("persona_refresh")
        reward = AtlasRewardBreakdown(score=0.8)
        evaluation = StepEvaluation(validation={}, reward=reward)
        self.result = Result(
            final_answer="learning complete",
            plan=Plan(steps=[]),
            step_results=[
                StepResult(
                    step_id=1,
                    trace="trace",
                    output="final output",
                    evaluation=evaluation,
                    attempts=2,
                    metadata={"persona_target": "student_executor", "reason": "Initial attempt lacked detail."},
                )
            ],
        )

    async def arun(self, task: str) -> Result:
        context = ExecutionContext.get()
        context.metadata["steps"] = {
            1: {
                "attempts": [
                    {
                        "attempt": 1,
                        "evaluation": {"reward": {"score": 0.3}},
                        "status": "retry",
                        "reason": "Too vague.",
                    },
                    {
                        "attempt": 2,
                        "evaluation": {"reward": {"score": 0.8}},
                        "status": "ok",
                    },
                ],
                "guidance": ["Provide more numerical detail."],
            }
        }
        if self._persona_refresh is not None:
            await self._persona_refresh()
        return self.result


async def test_persona_learning_engine_generates_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
    storage_config = StorageConfig(database_url=dsn, min_connections=1, max_connections=2)
    database = Database(storage_config)
    await database.connect()
    await _apply_schema(database)
    tenant_id = "tenant-learning"
    agent_name = "learning-agent"
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM persona_memory_usage USING persona_memory WHERE persona_memory_usage.memory_id = persona_memory.memory_id AND persona_memory.tenant_id = $1",
            tenant_id,
        )
        await connection.execute("DELETE FROM persona_memory WHERE tenant_id = $1", tenant_id)
    get_cache().clear()
    monkeypatch.delenv("ATLAS_PERSONA_MEMORY_CACHE_DISABLED", raising=False)
    agent_config = AdapterConfig(type=AdapterType.HTTP, name=agent_name, system_prompt="Base prompt", tools=[])
    shared_config = type(
        "ConfigWrapper",
        (),
        {
            "agent": agent_config,
            "student": StudentConfig(prompts=StudentPrompts(planner="{base_prompt}", executor="{base_prompt}", synthesizer="{base_prompt}")),
            "teacher": TeacherConfig(llm=LLMParameters(model="model")),
            "orchestration": OrchestrationConfig(max_retries=1, step_timeout_seconds=10, rim_guidance_tag="tag", emit_intermediate_steps=True),
            "rim": RIMConfig(
                small_model=LLMParameters(model="stub"),
                large_model=LLMParameters(model="arbiter"),
                active_judges={"process": True},
                variance_threshold=1.0,
                uncertainty_threshold=1.0,
                parallel_workers=1,
            ),
            "storage": storage_config,
            "prompt_rewrite": None,
            "metadata": {"persona_memory": {}},
        },
    )()

    context = ExecutionContext.get()
    context.reset()
    context.metadata["session_metadata"] = {"tenant_id": tenant_id, "tags": ["trial"]}
    fingerprint_inputs = extract_fingerprint_inputs("seed-task", shared_config, context)
    fingerprint = build_fingerprint(fingerprint_inputs)
    context.reset()

    active_memory_id = uuid4()
    await database.create_persona_memory(
        {
            "memory_id": active_memory_id,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "persona": "student_executor",
            "trigger_fingerprint": fingerprint,
            "instruction": {"append": "Provide extra numerical detail."},
            "source_session_id": None,
            "reward_snapshot": {"score": 0.3},
            "retry_count": 2,
            "metadata": {"tags": ["seed"], "helpful_count": 1, "harmful_count": 0, "neutral_count": 0, "last_reward": 0.3},
            "status": "active",
        }
    )

    await database.disconnect()

    monkeypatch.setattr(core, "load_config", lambda path: shared_config)
    monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
    monkeypatch.setattr(core, "Student", StubStudent)
    monkeypatch.setattr(core, "Teacher", StubTeacher)
    monkeypatch.setattr(core, "Evaluator", StubEvaluator)
    monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: FailingThenRecoveringOrchestrator(*args, **kwargs))

    task_name = "learning-engine-task"

    async def _run_and_inspect() -> tuple[Result, List[str]]:
        run_start = time.perf_counter()
        result = await core.arun(task_name, "config.yaml", session_metadata={"tenant_id": tenant_id, "tags": ["trial"]})
        duration_ms = (time.perf_counter() - run_start) * 1000.0
        print(f"Learning engine run completed in {duration_ms:.2f} ms")
        context = ExecutionContext.get()
        candidates = context.metadata.get("new_persona_candidates", [])
        return result, candidates

    result, candidate_ids = await _run_and_inspect()
    assert result.final_answer == "learning complete"
    assert len(candidate_ids) == 1
    candidate_uuid = UUID(candidate_ids[0])

    verification_db = Database(storage_config)
    await verification_db.connect()
    sessions = await verification_db.fetch_sessions(limit=20)
    session_entry = next(session for session in sessions if session["task"] == task_name)
    session_id = session_entry["id"]
    pool = verification_db._require_pool()  # noqa: SLF001 - integration test assertion
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT agent_name, tenant_id, persona, trigger_fingerprint, instruction, reward_snapshot, retry_count, source_session_id, status "
            "FROM persona_memory WHERE memory_id = $1",
            candidate_uuid,
        )
        usage_row = await connection.fetchrow(
            "SELECT reward, retry_count FROM persona_memory_usage WHERE memory_id = $1 ORDER BY id DESC LIMIT 1",
            active_memory_id,
        )
    await verification_db.disconnect()

    assert row is not None
    assert usage_row is not None
    reward_payload = Database._deserialize_json(usage_row["reward"])
    assert reward_payload is not None
    assert pytest.approx(reward_payload.get("score", 0.0), rel=1e-3) == 0.8
    assert usage_row["retry_count"] == 2

    assert row["agent_name"] == agent_name
    assert row["tenant_id"] == tenant_id
    assert row["persona"] == "student_executor"
    assert row["status"] == "candidate"
    assert row["source_session_id"] == session_id
    instruction_payload = row["instruction"]
    if isinstance(instruction_payload, str):
        instruction_payload = json.loads(instruction_payload)
    assert isinstance(instruction_payload, dict)
    assert instruction_payload.get("append") == "Provide more numerical detail."
    if "context" in instruction_payload:
        assert instruction_payload["context"] == "Too vague."
    reward_snapshot = row["reward_snapshot"]
    if isinstance(reward_snapshot, str):
        reward_snapshot = json.loads(reward_snapshot)
    assert isinstance(reward_snapshot, dict)
    assert reward_snapshot.get("reward", {}).get("score", reward_snapshot.get("score")) is not None
    assert row["retry_count"] == 2

    # Second run should reuse the same candidate (deduplicated)
    result_second, candidate_ids_second = await _run_and_inspect()
    assert result_second.final_answer == "learning complete"
    assert candidate_ids_second == []

    verification_db = Database(storage_config)
    await verification_db.connect()
    pool = verification_db._require_pool()  # noqa: SLF001 - integration test assertion
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT memory_id FROM persona_memory WHERE tenant_id = $1 AND persona = $2 AND status = 'candidate'",
            tenant_id,
            "student_executor",
        )
    await verification_db.disconnect()
    assert [record["memory_id"] for record in rows] == [candidate_uuid]
