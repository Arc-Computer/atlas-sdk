"""Integration tests for persona memory promotion, caps, and compaction."""

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
from atlas.runtime.persona_memory import get_cache
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Result

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


def _schema_statements() -> List[str]:
    schema_path = Path(__file__).resolve().parents[2] / "atlas" / "runtime" / "storage" / "schema.sql"
    return [
        statement.strip()
        for statement in schema_path.read_text().split(";")
        if statement.strip()
    ]


async def _apply_schema(database: Database) -> None:
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    statements = _schema_statements()
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


class NoOpOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        self._persona_refresh = kwargs.get("persona_refresh")
        self.result = Result(final_answer="promotion complete", plan=Plan(steps=[]), step_results=[])

    async def arun(self, task: str) -> Result:
        if self._persona_refresh is not None:
            await self._persona_refresh()
        ExecutionContext.get().metadata.setdefault("steps", {})
        return self.result


async def test_persona_promotion_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
    storage_config = StorageConfig(database_url=dsn, min_connections=1, max_connections=2)
    database = Database(storage_config)
    await database.connect()
    await _apply_schema(database)

    tenant_id = "tenant-promotion"
    agent_name = "promotion-agent"

    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM persona_memory_usage USING persona_memory WHERE persona_memory_usage.memory_id = persona_memory.memory_id AND persona_memory.tenant_id = $1",
            tenant_id,
        )
        await connection.execute("DELETE FROM persona_memory WHERE tenant_id = $1", tenant_id)

    # Construct config and fingerprint
    adapter_config = AdapterConfig(type=AdapterType.HTTP, name=agent_name, system_prompt="Promotion base prompt", tools=[])
    runtime_config = type(
        "ConfigWrapper",
        (),
        {
            "agent": adapter_config,
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
            "metadata": {
                "persona_memory": {
                    "promotion_samples": 2,
                    "promotion_threshold": 0.05,
                    "persona_caps": {"student": 2},
                }
            },
        },
    )()

    # Derive fingerprint
    context = ExecutionContext.get()
    context.reset()
    context.metadata["session_metadata"] = {"tenant_id": tenant_id, "tags": ["promotion"]}
    from atlas.runtime.persona_memory import extract_fingerprint_inputs, build_fingerprint  # noqa: WPS433

    fingerprint_inputs = extract_fingerprint_inputs("seed-task", runtime_config, context)
    fingerprint = build_fingerprint(fingerprint_inputs)
    context.reset()

    candidate_id = uuid4()
    await database.create_persona_memory(
        {
            "memory_id": candidate_id,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "persona": "student",
            "trigger_fingerprint": fingerprint,
            "instruction": {"append": "Provide extra quantitative detail."},
            "source_session_id": None,
            "reward_snapshot": {"score": 0.3},
            "retry_count": 2,
            "metadata": {"tags": ["candidate"], "helpful_count": 0, "harmful_count": 0, "neutral_count": 0},
            "status": "candidate",
        }
    )
    seed_session = await database.create_session("promotion-seed")
    await database.log_persona_memory_usage(candidate_id, seed_session, reward={"score": 0.4}, retries=2, mode="coach")
    seed_session_2 = await database.create_session("promotion-seed-2")
    await database.log_persona_memory_usage(candidate_id, seed_session_2, reward={"score": 0.9}, retries=1, mode="auto")

    active_duplicate_1 = uuid4()
    active_duplicate_2 = uuid4()
    duplicate_instruction = {"append": "Ensure clarity in responses."}
    await database.create_persona_memory(
        {
            "memory_id": active_duplicate_1,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "persona": "student",
            "trigger_fingerprint": fingerprint,
            "instruction": duplicate_instruction,
            "source_session_id": None,
            "reward_snapshot": {"score": 0.5},
            "retry_count": 1,
            "metadata": {"tags": ["seed"], "helpful_count": 1, "harmful_count": 0, "neutral_count": 0, "last_reward": 0.5},
            "status": "active",
        }
    )
    await database.create_persona_memory(
        {
            "memory_id": active_duplicate_2,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "persona": "student",
            "trigger_fingerprint": fingerprint,
            "instruction": duplicate_instruction,
            "source_session_id": None,
            "reward_snapshot": {"score": 0.45},
            "retry_count": 1,
            "metadata": {"tags": ["seed"], "helpful_count": 1, "harmful_count": 0, "neutral_count": 0, "last_reward": 0.45},
            "status": "active",
        }
    )
    await database.disconnect()

    get_cache().clear()
    monkeypatch.delenv("ATLAS_PERSONA_MEMORY_CACHE_DISABLED", raising=False)
    monkeypatch.setattr(core, "load_config", lambda path: runtime_config)
    monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
    monkeypatch.setattr(core, "Student", StubStudent)
    monkeypatch.setattr(core, "Teacher", StubTeacher)
    monkeypatch.setattr(core, "Evaluator", StubEvaluator)
    monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: NoOpOrchestrator(*args, **kwargs))

    promotion_start = time.perf_counter()
    result = await core.arun(
        "persona-promotion-task",
        "config.yaml",
        session_metadata={"tenant_id": tenant_id, "tags": ["promotion"]},
    )
    elapsed_ms = (time.perf_counter() - promotion_start) * 1000.0
    print(f"Promotion workflow completed in {elapsed_ms:.2f} ms")

    assert result.final_answer == "promotion complete"

    verification_db = Database(storage_config)
    await verification_db.connect()
    pool = verification_db._require_pool()  # noqa: SLF001 - integration test assertion
    async with pool.acquire() as connection:
        candidate_row = await connection.fetchrow(
            "SELECT status, instruction FROM persona_memory WHERE memory_id = $1",
            candidate_id,
        )
        active_rows = await connection.fetch(
            "SELECT memory_id, status FROM persona_memory WHERE tenant_id = $1 AND persona = $2",
            tenant_id,
            "student",
        )
    await verification_db.disconnect()

    assert candidate_row is not None
    assert candidate_row["status"] == "active"

    statuses = {row["memory_id"]: row["status"] for row in active_rows}
    assert statuses.get(candidate_id) == "active"
    demoted_ids = [memory_id for memory_id, status in statuses.items() if memory_id != candidate_id and status != "active"]
    assert demoted_ids, "At least one duplicate should be demoted or replaced"

    promotion_metadata = ExecutionContext.get().metadata.get("persona_promotion_result")
    assert promotion_metadata is not None
    assert str(candidate_id) in promotion_metadata.get("promoted", [])
    for demoted in demoted_ids:
        assert str(demoted) in promotion_metadata.get("demoted", []) or str(demoted) in promotion_metadata.get("compacted", [])

    normalized_instruction = candidate_row["instruction"]
    if isinstance(normalized_instruction, str):
        normalized_instruction = json.loads(normalized_instruction)
    assert isinstance(normalized_instruction, dict)
