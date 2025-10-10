"""Integration tests for persona memory fingerprinting and caching."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import MethodType
from typing import Any, List
from uuid import uuid4

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas.config.models import AdapterConfig, AdapterType, StorageConfig, ToolDefinition
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.persona_memory import PersonaMemoryKey, build_fingerprint, extract_fingerprint_inputs, get_cache, is_cache_disabled
from atlas.runtime.persona_memory.fingerprint import FingerprintInputs
from atlas.runtime.storage.database import Database

pytestmark = pytest.mark.asyncio


def _apply_schema_sync() -> List[str]:
    schema_path = Path(__file__).resolve().parents[2] / "atlas" / "runtime" / "storage" / "schema.sql"
    return [
        statement.strip()
        for statement in schema_path.read_text().split(";")
        if statement.strip()
    ]


async def _apply_schema(database: Database) -> None:
    statements = _apply_schema_sync()
    pool = database._require_pool()  # noqa: SLF001 - integration test setup
    async with pool.acquire() as connection:
        for statement in statements:
            await connection.execute(statement)


def _create_config(agent_name: str, tools: List[str]) -> Any:
    tool_defs = [ToolDefinition(name=tool, description=f"{tool} tool") for tool in tools]
    adapter = AdapterConfig(type=AdapterType.HTTP, name=agent_name, system_prompt="Base prompt", tools=tool_defs)
    return type("Config", (), {"agent": adapter, "metadata": {"persona_memory": {}}})()


def _prepare_fingerprint(
    task: str,
    config: Any,
    execution_context: ExecutionContext,
) -> tuple[FingerprintInputs, str]:
    inputs = extract_fingerprint_inputs(task, config, execution_context)
    return inputs, build_fingerprint(inputs)


async def test_persona_memory_cache_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
    storage_config = StorageConfig(database_url=dsn, min_connections=1, max_connections=2)
    database = Database(storage_config)
    await database.connect()
    await _apply_schema(database)

    execution_context = ExecutionContext.get()
    execution_context.reset()
    tenant_id = "tenant-cache"
    execution_context.metadata["session_metadata"] = {"tenant_id": tenant_id, "tags": ["tag-b", "tag-a"]}
    execution_context.metadata["execution_mode"] = "stepwise"

    config = _create_config("cache-agent", ["search", "browse"])
    inputs, fingerprint = _prepare_fingerprint("cache-task", config, execution_context)

    memory_id = uuid4()
    await database.create_persona_memory(
        {
            "memory_id": memory_id,
            "agent_name": inputs.agent_name,
            "tenant_id": inputs.tenant_id,
            "persona": "student_planner",
            "trigger_fingerprint": fingerprint,
            "instruction": {"notes": ["first"]},
            "source_session_id": None,
            "reward_snapshot": {"score": 0.3},
            "retry_count": 0,
            "status": "active",
        }
    )

    cache = get_cache()
    cache.clear()
    fetch_calls = 0
    original_fetch = database.fetch_persona_memories

    async def tracking_fetch(self, agent_name, tenant, persona, trigger, statuses=None):
        nonlocal fetch_calls
        fetch_calls += 1
        return await original_fetch(agent_name, tenant, persona, trigger, statuses=statuses)

    database.fetch_persona_memories = MethodType(tracking_fetch, database)

    key = PersonaMemoryKey(
        agent_name=inputs.agent_name,
        tenant_id=inputs.tenant_id,
        fingerprint=fingerprint,
        persona="student_planner",
    )
    try:
        use_cache = not is_cache_disabled(config)
        start = time.perf_counter()
        first = await cache.get_or_load(database, key, ["active"], use_cache=use_cache)
        mid = time.perf_counter()
        second = await cache.get_or_load(database, key, ["active"], use_cache=use_cache)
        end = time.perf_counter()
        assert fetch_calls == 1
        assert first == second
        print(f"Caching path initial fetch {(mid - start) * 1000.0:.2f} ms, cached reuse {(end - mid) * 1000.0:.2f} ms")

        execution_context.metadata["session_metadata"]["tags"] = ["new-tag"]
        new_inputs, new_fingerprint = _prepare_fingerprint("cache-task", config, execution_context)
        await database.create_persona_memory(
            {
                "memory_id": uuid4(),
                "agent_name": new_inputs.agent_name,
                "tenant_id": new_inputs.tenant_id,
                "persona": "student_planner",
                "trigger_fingerprint": new_fingerprint,
                "instruction": {"notes": ["second"]},
                "source_session_id": None,
                "reward_snapshot": None,
                "retry_count": None,
                "status": "active",
            }
        )
        key_new = PersonaMemoryKey(
            agent_name=new_inputs.agent_name,
            tenant_id=new_inputs.tenant_id,
            fingerprint=new_fingerprint,
            persona="student_planner",
        )
        await cache.get_or_load(database, key_new, ["active"], use_cache=use_cache)
        assert fetch_calls == 2

        cache.clear()
        monkeypatch.setenv("ATLAS_PERSONA_MEMORY_CACHE_DISABLED", "1")
        use_cache_disabled = not is_cache_disabled(config)
        await cache.get_or_load(database, key, ["active"], use_cache=use_cache_disabled)
        await cache.get_or_load(database, key, ["active"], use_cache=use_cache_disabled)
        assert fetch_calls == 4
        print("Cache disabled path triggered direct database fetch twice.")
    finally:
        database.fetch_persona_memories = original_fetch
        await database.disconnect()
        execution_context.reset()
