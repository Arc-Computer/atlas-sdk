import json

import pytest

pytest.importorskip("asyncpg")

from atlas.config.models import StorageConfig
from atlas.runtime.models import IntermediateStep, IntermediateStepPayload, IntermediateStepType, StreamEventData
from atlas.runtime.models import InvocationNode
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Step, StepEvaluation, StepResult


class FakeConnection:
    def __init__(self):
        self.commands = []

    async def execute(self, sql, *args):
        self.commands.append((sql, args))
        if sql.startswith("SET"):
            return None
        return None

    async def fetchval(self, sql, *args):
        self.commands.append((sql, args))
        return 42

    async def executemany(self, sql, records):
        self.commands.append((sql, tuple(records)))

    async def fetch(self, sql, *args):
        self.commands.append((sql, args))
        return []

    async def fetchrow(self, sql, *args):
        self.commands.append((sql, args))
        return None


class FakeAcquire:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return FakeAcquire(self.connection)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_database_logs_plan_steps(monkeypatch):
    pool = FakePool()
    import asyncpg

    async def fake_create_pool(**_):
        return pool

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    config = StorageConfig(database_url="postgresql://stub", min_connections=1, max_connections=2, statement_timeout_seconds=1)
    database = Database(config)
    await database.connect()
    session_id = await database.create_session("task")
    plan = Plan(steps=[Step(id=1, description="desc", depends_on=[])])
    await database.log_plan(session_id, plan)
    reward = AtlasRewardBreakdown(score=1.0)
    evaluation = StepEvaluation(validation={}, reward=reward)
    step_result = StepResult(step_id=1, trace="trace", output="output", evaluation=evaluation, attempts=1)
    await database.log_step_result(session_id, step_result)
    await database.log_step_attempts(session_id, 1, [{"attempt": 1, "evaluation": {"score": 1.0}}])
    await database.log_guidance(session_id, 1, ["note"])
    event = IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(function_id="root", function_name="root"),
        payload=IntermediateStepPayload(
            event_type=IntermediateStepType.WORKFLOW_START,
            name="test",
            data=StreamEventData(input={"sample": True}),
        ),
    )
    await database.log_intermediate_step(session_id, event)
    await database.finalize_session(session_id, "answer", "succeeded")
    commands = pool.connection.commands
    assert any("INSERT INTO plans" in command[0] for command in commands)
    assert any("INSERT INTO step_results" in command[0] for command in commands)
    assert any("INSERT INTO step_attempts" in command[0] for command in commands)
    assert any("INSERT INTO guidance_notes" in command[0] for command in commands)
    assert any("INSERT INTO trajectory_events" in command[0] for command in commands)
    step_insert = next(cmd for cmd in commands if "INSERT INTO step_results" in cmd[0])
    assert "metadata" in step_insert[0].lower()
    await database.disconnect()


@pytest.mark.asyncio
async def test_database_logs_discovery_run(monkeypatch):
    pool = FakePool()
    import asyncpg

    async def fake_create_pool(**_):
        return pool

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    config = StorageConfig(database_url="postgresql://stub", min_connections=1, max_connections=2, statement_timeout_seconds=1)
    database = Database(config)
    await database.connect()
    payload = {"plan": {"steps": []}, "telemetry": {"events": []}}
    metadata = {"capabilities": {"telemetry_agent_emitted": True}}
    run_id = await database.log_discovery_run(
        project_root="/tmp/project",
        task="Telemetry integration test",
        payload=payload,
        metadata=metadata,
        source="discovery",
    )
    assert run_id == 42
    commands = pool.connection.commands
    discovery_insert = next(cmd for cmd in commands if "INSERT INTO discovery_runs" in cmd[0])
    assert "discovery_runs" in discovery_insert[0]
    args = discovery_insert[1]
    assert args[0] == "/tmp/project"
    assert args[1] == "Telemetry integration test"
    assert json.loads(args[3])["plan"] == {"steps": []}
    assert json.loads(args[4])["capabilities"]["telemetry_agent_emitted"] is True
    await database.disconnect()
