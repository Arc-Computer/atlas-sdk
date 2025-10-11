"""Integration tests for persona memory persistence.

Commands to reproduce manually:
    docker compose -f docker/docker-compose.yaml up -d postgres
    export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas_arc_demo
    pytest tests/integration/test_persona_memory.py
    pytest tests/integration/test_core.py
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas import core
from atlas.config.models import AdapterConfig, AdapterType, LLMParameters, OrchestrationConfig, RIMConfig, StorageConfig, StudentConfig, StudentPrompts, TeacherConfig
from atlas.prompts import RewrittenStudentPrompts, RewrittenTeacherPrompts
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Result

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


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


class StubOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        self._persona_refresh = kwargs.get("persona_refresh")
        self.result = Result(final_answer="persona run complete", plan=Plan(steps=[]), step_results=[])

    async def arun(self, task: str) -> Result:
        if self._persona_refresh is not None:
            await self._persona_refresh()
        return self.result


async def _apply_schema(database: Database) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "atlas" / "runtime" / "storage" / "schema.sql"
    statements = [
        statement.strip()
        for statement in schema_path.read_text().split(";")
        if statement.strip()
    ]
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        for statement in statements:
            await connection.execute(statement)


async def _wait_for_postgres(dsn: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = await asyncpg.connect(dsn=dsn)
        except Exception:  # pragma: no cover - retry loop
            await asyncio.sleep(0.5)
        else:
            await conn.close()
            return
    raise RuntimeError("Postgres did not become ready within the allotted time.")


async def test_persona_memory_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
    os.environ["STORAGE__DATABASE_URL"] = dsn
    await _wait_for_postgres(dsn)

    storage_config = StorageConfig(database_url=dsn, min_connections=1, max_connections=2)
    database = Database(storage_config)
    await database.connect()
    try:
        await _apply_schema(database)
        workflow_start = time.perf_counter()
        session_id = await database.create_session("persona-memory-test", metadata={"purpose": "integration"})
        memory_id = uuid4()
        agent_name = "memory-agent"
        tenant_id = "tenant-123"
        persona = "analyst"
        fingerprint = f"fp-{uuid4()}"
        persona_record: dict[str, Any] = {
            "memory_id": memory_id,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "persona": persona,
            "trigger_fingerprint": fingerprint,
            "instruction": {"steps": ["observe", "report"]},
            "source_session_id": session_id,
            "reward_snapshot": None,
            "retry_count": None,
            "status": "candidate",
        }
        await database.create_persona_memory(persona_record)

        fetched = await database.fetch_persona_memories(agent_name, tenant_id, persona, fingerprint)
        assert len(fetched) == 1
        first = fetched[0]
        assert first["memory_id"] == memory_id
        assert first["instruction"] == {"steps": ["observe", "report"]}
        assert first["status"] == "candidate"
        assert first["reward_snapshot"] is None

        await database.update_persona_memory_status(
            memory_id,
            "active",
            reward_snapshot={"score": 0.75},
            retry_count=3,
        )
        active_records = await database.fetch_persona_memories(agent_name, tenant_id, persona, fingerprint, statuses=["active"])
        assert len(active_records) == 1
        active = active_records[0]
        assert active["status"] == "active"
        assert active["reward_snapshot"] == {"score": 0.75}
        assert active["retry_count"] == 3
        assert active["updated_at"] >= active["created_at"]

        await database.log_persona_memory_usage(memory_id, session_id, reward={"score": 0.9}, retries=2)
        pool = database._require_pool()  # noqa: SLF001 - integration test assertion
        async with pool.acquire() as connection:
            usage_row = await connection.fetchrow(
                "SELECT memory_id, session_id, reward, retry_count FROM persona_memory_usage WHERE memory_id = $1",
                memory_id,
            )
        assert usage_row is not None
        reward_payload = Database._deserialize_json(usage_row["reward"])
        assert usage_row["session_id"] == session_id
        assert reward_payload == {"score": 0.9}
        assert usage_row["retry_count"] == 2

        workflow_elapsed_ms = (time.perf_counter() - workflow_start) * 1000.0
        print(f"Persona memory workflow completed in {workflow_elapsed_ms:.2f} ms")
    finally:
        await database.disconnect()

    run_start = time.perf_counter()

    atlas_config = type(
        "Config",
        (),
        {
            "agent": AdapterConfig(type=AdapterType.HTTP, name="stub", system_prompt="Base", tools=[]),
            "student": StudentConfig(prompts=StudentPrompts(planner="{base_prompt}", executor="{base_prompt}", synthesizer="{base_prompt}")),
            "teacher": TeacherConfig(llm=LLMParameters(model="model")),
            "orchestration": OrchestrationConfig(max_retries=0, step_timeout_seconds=10, rim_guidance_tag="tag", emit_intermediate_steps=True),
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
        },
    )()

    monkeypatch.setattr(core, "load_config", lambda path: atlas_config)
    monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
    monkeypatch.setattr(core, "Student", StubStudent)
    monkeypatch.setattr(core, "Teacher", StubTeacher)
    monkeypatch.setattr(core, "Evaluator", StubEvaluator)
    monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: StubOrchestrator(*args, **kwargs))
    monkeypatch.setattr(
        core,
        "build_student_prompts",
        lambda *_args, **_kwargs: RewrittenStudentPrompts("planner", "executor", "synth"),
    )
    monkeypatch.setattr(
        core,
        "build_teacher_prompts",
        lambda *_args, **_kwargs: RewrittenTeacherPrompts("plan", "validate", "guide"),
    )

    result = await core.arun("persona-core-run", "config.yaml")
    assert result.final_answer == "persona run complete"

    regression_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
    print(f"Atlas core run completed in {regression_elapsed_ms:.2f} ms")

    verification_db = Database(storage_config)
    await verification_db.connect()
    try:
        sessions = await verification_db.fetch_sessions(limit=5)
        assert any(session["task"] == "persona-core-run" for session in sessions)
    finally:
        await verification_db.disconnect()
