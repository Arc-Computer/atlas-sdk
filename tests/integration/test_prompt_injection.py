"""Integration tests covering runtime prompt injection with persona memories."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas import core
from atlas.config.models import AdapterConfig, AdapterType, LLMParameters, OrchestrationConfig, RIMConfig, StorageConfig, StudentConfig, StudentPrompts, TeacherConfig, ToolDefinition
from atlas.prompts import RewrittenStudentPrompts, RewrittenTeacherPrompts
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.persona_memory import build_fingerprint, extract_fingerprint_inputs, get_cache
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Result

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]


def _apply_schema_statements() -> List[str]:
    schema_path = Path(__file__).resolve().parents[2] / "atlas" / "runtime" / "storage" / "schema.sql"
    return [
        statement.strip()
        for statement in schema_path.read_text().split(";")
        if statement.strip()
    ]


async def _apply_schema(database: Database) -> None:
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    statements = _apply_schema_statements()
    async with pool.acquire() as connection:
        for statement in statements:
            await connection.execute(statement)


class StubAdapter:
    async def ainvoke(self, prompt: str, metadata=None):
        return "{}"


class CapturingStudent:
    instances: List["CapturingStudent"] = []

    def __init__(self, adapter, adapter_config, student_config, student_prompts: RewrittenStudentPrompts) -> None:
        self.prompts = student_prompts
        CapturingStudent.instances.append(self)

    def update_prompts(self, student_prompts: RewrittenStudentPrompts) -> None:
        self.prompts = student_prompts


class CapturingTeacher:
    instances: List["CapturingTeacher"] = []

    def __init__(self, config: TeacherConfig, prompts: RewrittenTeacherPrompts) -> None:
        self.prompts = prompts
        CapturingTeacher.instances.append(self)

    def update_prompts(self, prompts: RewrittenTeacherPrompts) -> None:
        self.prompts = prompts


class StubEvaluator:
    def __init__(self, *args, **kwargs) -> None:
        pass


class StubOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        self._persona_refresh = kwargs.get("persona_refresh")
        self.result = Result(final_answer="prompt injection complete", plan=Plan(steps=[]), step_results=[])

    async def arun(self, task: str) -> Result:
        if self._persona_refresh is not None:
            await self._persona_refresh()
        return self.result


async def test_prompt_injection_and_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
    storage_config = StorageConfig(database_url=dsn, min_connections=1, max_connections=2)
    database = Database(storage_config)
    await database.connect()
    await _apply_schema(database)
    cache = get_cache()
    cache.clear()
    tenant_id = "tenant-persona-injection"
    agent_name = "persona-agent"
    tools = [
        ToolDefinition(name="search", description="search tool"),
        ToolDefinition(name="lookup", description="lookup tool"),
    ]
    # Compute fingerprint for seeding
    context = ExecutionContext.get()
    context.reset()
    context.metadata["session_metadata"] = {"tenant_id": tenant_id, "tags": ["beta", "alpha"]}
    context.metadata["execution_mode"] = "stepwise"
    adapter_config = AdapterConfig(type=AdapterType.HTTP, name=agent_name, system_prompt="Base persona prompt", tools=tools)
    atlas_config = type("Config", (), {"agent": adapter_config, "metadata": {"persona_memory": {}}})()
    fingerprint_inputs = extract_fingerprint_inputs("seed-task", atlas_config, context)
    fingerprint = build_fingerprint(fingerprint_inputs)
    context.reset()

    persona_payloads: Dict[str, List[Dict[str, Any]]] = {
        "student_planner": [
            {"instruction": {"prepend": "Planner persona preface."}, "status": "active"},
            {"instruction": "Always include citations.", "status": "active"},
        ],
        "student_executor": [
            {"instruction": {"append": "Report tool usage explicitly."}, "status": "active"},
            {"instruction": {"append": "Trial persona augmentation."}, "status": "candidate"},
        ],
        "student_synthesizer": [
            {"instruction": {"replace": "Provide a concise bullet summary reflecting persona memory."}, "status": "active"},
        ],
        "teacher_plan_review": [
            {"instruction": {"prepend": "Demand risk assessment before approval."}, "status": "active"},
        ],
        "teacher_validation": [
            {"instruction": {"append": "Request additional evidence when uncertain."}, "status": "active"},
        ],
        "teacher_guidance": [
            {"instruction": "Offer constructive encouragement in guidance.", "status": "active"},
        ],
    }
    persona_memory_ids: Dict[str, List[Dict[str, Any]]] = {persona: [] for persona in persona_payloads}

    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM persona_memory_usage USING persona_memory WHERE persona_memory_usage.memory_id = persona_memory.memory_id AND persona_memory.tenant_id = $1",
            tenant_id,
        )
        await connection.execute("DELETE FROM persona_memory WHERE tenant_id = $1", tenant_id)
    for persona, instructions in persona_payloads.items():
        for payload in instructions:
            memory_id = uuid4()
            persona_memory_ids[persona].append({"memory_id": memory_id, "status": payload.get("status", "active")})
            await database.create_persona_memory(
                {
                    "memory_id": memory_id,
                    "agent_name": agent_name,
                    "tenant_id": tenant_id,
                    "persona": persona,
                    "trigger_fingerprint": fingerprint,
                    "instruction": payload["instruction"],
                    "source_session_id": None,
                    "reward_snapshot": None,
                    "retry_count": None,
                    "status": payload.get("status", "active"),
                }
            )
    await database.disconnect()

    CapturingStudent.instances.clear()
    CapturingTeacher.instances.clear()
    monkeypatch.delenv("ATLAS_PERSONA_MEMORY_CACHE_DISABLED", raising=False)
    monkeypatch.setattr(core, "load_config", lambda path: type("ConfigWrapper", (), {
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
        "metadata": {"persona_memory": {}},
    })())
    monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
    monkeypatch.setattr(core, "Student", CapturingStudent)
    monkeypatch.setattr(core, "Teacher", CapturingTeacher)
    monkeypatch.setattr(core, "Evaluator", StubEvaluator)
    monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: StubOrchestrator(*args, **kwargs))

    task_name = "prompt-injection-task"
    run_start = time.perf_counter()
    result = await core.arun(task_name, "config.yaml", session_metadata={"tenant_id": tenant_id, "tags": ["alpha", "beta"]})
    duration_ms = (time.perf_counter() - run_start) * 1000.0
    print(f"Prompt injection run completed in {duration_ms:.2f} ms")
    assert result.final_answer == "prompt injection complete"

    assert CapturingStudent.instances, "Student should be instantiated"
    assert CapturingTeacher.instances, "Teacher should be instantiated"
    student_prompts = CapturingStudent.instances[-1].prompts
    teacher_prompts = CapturingTeacher.instances[-1].prompts

    assert student_prompts.planner.lstrip().startswith("Planner persona preface.")
    assert "Always include citations." in student_prompts.planner
    assert "Report tool usage explicitly." in student_prompts.executor
    assert "Trial persona augmentation." in student_prompts.executor
    assert student_prompts.synthesizer.strip() == "Provide a concise bullet summary reflecting persona memory."
    assert teacher_prompts.plan_review.lstrip().startswith("Demand risk assessment before approval.")
    assert teacher_prompts.validation.rstrip().endswith("Request additional evidence when uncertain.")
    assert "Offer constructive encouragement in guidance." in teacher_prompts.guidance

    context = ExecutionContext.get()
    applied = context.metadata.get("applied_persona_memories")
    assert applied is not None
    for persona, expected_entries in persona_memory_ids.items():
        actual = applied.get(persona)
        assert isinstance(actual, list)
        normalised = [{"memory_id": entry.get("memory_id"), "status": entry.get("status") or "active"} for entry in actual]
        assert normalised == expected_entries

    verification_db = Database(storage_config)
    await verification_db.connect()
    sessions = await verification_db.fetch_sessions(limit=10)
    session_entry = next(session for session in sessions if session["task"] == task_name)
    session_id = session_entry["id"]
    pool = verification_db._require_pool()  # noqa: SLF001 - integration test assertion
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT memory_id FROM persona_memory_usage WHERE session_id = $1 ORDER BY memory_id",
            session_id,
        )
    await verification_db.disconnect()
    expected_ids = sorted({entry["memory_id"] for entries in persona_memory_ids.values() for entry in entries})
    logged_ids = sorted(row["memory_id"] for row in rows)
    assert logged_ids == expected_ids
