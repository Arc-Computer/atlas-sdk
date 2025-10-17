"""Asynchronous PostgreSQL persistence layer."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

try:
    from importlib import resources as importlib_resources
except ImportError:  # pragma: no cover
    import importlib_resources  # type: ignore

try:
    import asyncpg
    _ASYNCPG_ERROR = None
except ModuleNotFoundError as exc:
    asyncpg = None
    _ASYNCPG_ERROR = exc

from atlas.config.models import StorageConfig
from atlas.runtime.models import IntermediateStep
from atlas.types import Plan
from atlas.types import StepResult
from atlas.runtime.schema import AtlasRewardBreakdown


class Database:
    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._schema_sql: str | None = None
        self._schema_initialized: bool = False

    async def connect(self) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for database persistence") from _ASYNCPG_ERROR
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._config.database_url,
                min_size=self._config.min_connections,
                max_size=self._config.max_connections,
                statement_cache_size=0,
            )
            async with self._pool.acquire() as connection:
                await connection.execute(f"SET statement_timeout = {int(self._config.statement_timeout_seconds * 1000)}")
                await self._initialize_schema(connection)

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_session(self, task: str, metadata: Dict[str, Any] | None = None) -> int:
        pool = self._require_pool()
        serialized_metadata = self._serialize_json(metadata) if metadata else None
        async with pool.acquire() as connection:
            return await connection.fetchval(
                "INSERT INTO sessions(task, metadata) VALUES ($1, $2) RETURNING id",
                task,
                serialized_metadata,
            )

    async def log_plan(self, session_id: int, plan: Plan) -> None:
        pool = self._require_pool()
        serialized_plan = self._serialize_json(plan.model_dump())
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO plans(session_id, plan) VALUES ($1, $2)"
                " ON CONFLICT (session_id) DO UPDATE SET plan = EXCLUDED.plan",
                session_id,
                serialized_plan,
            )

    async def log_step_result(self, session_id: int, result: StepResult) -> None:
        pool = self._require_pool()
        if hasattr(result.evaluation, "to_dict"):
            evaluation_payload = result.evaluation.to_dict()
        else:
            evaluation_payload = result.evaluation
        serialized_evaluation = self._serialize_json(evaluation_payload)
        serialized_metadata = self._serialize_json(result.metadata) if getattr(result, "metadata", None) else None
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO step_results(session_id, step_id, trace, output, evaluation, attempts, metadata)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7)"
                " ON CONFLICT (session_id, step_id) DO UPDATE SET"
                " trace = EXCLUDED.trace, output = EXCLUDED.output, evaluation = EXCLUDED.evaluation,"
                " attempts = EXCLUDED.attempts, metadata = EXCLUDED.metadata",
                session_id,
                result.step_id,
                result.trace,
                result.output,
                serialized_evaluation,
                result.attempts,
                serialized_metadata,
            )

    async def log_step_attempts(
        self,
        session_id: int,
        step_id: int,
        attempts: Iterable[Dict[str, Any]],
    ) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM step_attempts WHERE session_id = $1 AND step_id = $2",
                session_id,
                step_id,
            )
            records = [
                (session_id, step_id, attempt.get("attempt", index + 1), self._serialize_json(attempt.get("evaluation")))
                for index, attempt in enumerate(attempts)
            ]
            if records:
                await connection.executemany(
                    "INSERT INTO step_attempts(session_id, step_id, attempt, evaluation) VALUES ($1, $2, $3, $4)",
                    records,
                )

    async def log_intermediate_step(self, session_id: int, event: IntermediateStep) -> None:
        pool = self._require_pool()
        serialized_event = self._serialize_json(event.model_dump())
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO trajectory_events(session_id, event) VALUES ($1, $2)",
                session_id,
                serialized_event,
            )

    async def log_guidance(self, session_id: int, step_id: int, notes: Iterable[str]) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM guidance_notes WHERE session_id = $1 AND step_id = $2",
                session_id,
                step_id,
            )
            records = [(session_id, step_id, index, note) for index, note in enumerate(notes, start=1)]
            if records:
                await connection.executemany(
                    "INSERT INTO guidance_notes(session_id, step_id, sequence, note) VALUES ($1, $2, $3, $4)",
                    records,
                )

    async def finalize_session(self, session_id: int, final_answer: str, status: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE sessions SET status = $1, final_answer = $2, completed_at = NOW() WHERE id = $3",
                status,
                final_answer,
                session_id,
            )

    async def log_session_reward(
        self,
        session_id: int,
        reward: AtlasRewardBreakdown | Dict[str, Any] | None,
        student_learning: Optional[str],
        teacher_learning: Optional[str],
    ) -> None:
        pool = self._require_pool()
        serialized_reward = self._serialize_json(reward.to_dict() if hasattr(reward, "to_dict") else reward) if reward else None
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE sessions SET reward = $1, student_learning = $2, teacher_learning = $3 WHERE id = $4",
                serialized_reward,
                student_learning,
                teacher_learning,
                session_id,
            )

    async def fetch_sessions(self, limit: int = 50, offset: int = 0) -> List[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, task, status, metadata, final_answer, reward, student_learning, teacher_learning, created_at, completed_at"
                " FROM sessions ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
        return [dict(row) for row in rows]

    async def fetch_session(self, session_id: int) -> dict[str, Any] | None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT id, task, status, metadata, final_answer, reward, student_learning, teacher_learning, created_at, completed_at"
                " FROM sessions WHERE id = $1",
                session_id,
            )
            if row is None:
                return None
            plan_row = await connection.fetchrow(
                "SELECT plan FROM plans WHERE session_id = $1",
                session_id,
            )
        session = dict(row)
        session["plan"] = plan_row["plan"] if plan_row else None
        return session

    async def _initialize_schema(self, connection: "asyncpg.connection.Connection") -> None:
        if self._schema_initialized:
            return
        if not getattr(self._config, "apply_schema_on_connect", True):
            return
        if self._schema_sql is None:
            resource = importlib_resources.files("atlas.runtime.storage").joinpath("schema.sql")
            self._schema_sql = resource.read_text(encoding="utf-8")
        await connection.execute(self._schema_sql)
        self._schema_initialized = True

    async def fetch_session_steps(self, session_id: int) -> List[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            step_rows = await connection.fetch(
                "SELECT step_id, trace, output, evaluation, attempts, metadata"
                " FROM step_results WHERE session_id = $1 ORDER BY step_id",
                session_id,
            )
            attempt_rows = await connection.fetch(
                "SELECT step_id, attempt, evaluation"
                " FROM step_attempts WHERE session_id = $1 ORDER BY step_id, attempt",
                session_id,
            )
            guidance_rows = await connection.fetch(
                "SELECT step_id, sequence, note"
                " FROM guidance_notes WHERE session_id = $1 ORDER BY step_id, sequence",
                session_id,
            )
        attempts_by_step: dict[int, list[dict[str, Any]]] = {}
        for row in attempt_rows:
            attempts_by_step.setdefault(row["step_id"], []).append(
                {"attempt": row["attempt"], "evaluation": row["evaluation"]}
            )
        guidance_by_step: dict[int, list[str]] = {}
        for row in guidance_rows:
            guidance_by_step.setdefault(row["step_id"], []).append(row["note"])
        results: list[dict[str, Any]] = []
        for row in step_rows:
            step_id = row["step_id"]
            results.append(
                {
                    "step_id": step_id,
                    "trace": row["trace"],
                    "output": row["output"],
                    "evaluation": row["evaluation"],
                    "attempts": row["attempts"],
                    "metadata": row["metadata"],
                    "attempt_details": attempts_by_step.get(step_id, []),
                    "guidance_notes": guidance_by_step.get(step_id, []),
                }
                )
        return results

    async def fetch_learning_history(self, learning_key: str) -> List[dict[str, Any]]:
        if not learning_key:
            return []
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT reward, student_learning, teacher_learning, created_at, completed_at"
                " FROM sessions"
                " WHERE metadata->>'learning_key' = $1 AND reward IS NOT NULL"
                " ORDER BY created_at ASC",
                learning_key,
            )
        history: List[dict[str, Any]] = []
        for row in rows:
            history.append(
                {
                    "reward": self._deserialize_json(row["reward"]),
                    "student_learning": row.get("student_learning"),
                    "teacher_learning": row.get("teacher_learning"),
                    "created_at": row.get("created_at"),
                    "completed_at": row.get("completed_at"),
                }
            )
        return history

    async def fetch_trajectory_events(self, session_id: int, limit: int = 200) -> List[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, event, created_at FROM trajectory_events"
                " WHERE session_id = $1 ORDER BY id DESC LIMIT $2",
                session_id,
                limit,
            )
        return [dict(row) for row in rows]


    async def update_session_metadata(self, session_id: int, metadata: Dict[str, Any]) -> None:
        """Replace metadata payload for a session."""
        pool = self._require_pool()
        payload = self._serialize_json(metadata) if metadata is not None else None
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE sessions SET metadata = $2 WHERE id = $1",
                session_id,
                payload,
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database connection has not been established")
        return self._pool

    @staticmethod
    def _serialize_json(data: Any) -> str | None:
        """Convert data to JSON string for asyncpg JSONB columns."""
        if data is None:
            return None
        try:
            return json.dumps(data, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(data))

    @staticmethod
    def _deserialize_json(data: Any) -> Any:
        """Convert JSON payloads retrieved from the database into Python objects."""
        if data is None or isinstance(data, (dict, list)):
            return data
        try:
            return json.loads(data)
        except (TypeError, ValueError):
            return data
