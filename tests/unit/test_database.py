import pytest

pytest.importorskip("asyncpg")

from atlas.config.models import StorageConfig
from atlas.data_models.intermediate_step import IntermediateStep, IntermediateStepPayload, IntermediateStepType, StreamEventData
from atlas.data_models.invocation_node import InvocationNode
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.storage.database import Database
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
    await database.disconnect()
