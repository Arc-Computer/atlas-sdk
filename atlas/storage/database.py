"""Asynchronous PostgreSQL persistence layer."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable

try:
    import asyncpg
    _ASYNCPG_ERROR = None
except ModuleNotFoundError as exc:
    asyncpg = None
    _ASYNCPG_ERROR = exc

from atlas.config.models import StorageConfig
from atlas.data_models.intermediate_step import IntermediateStep
from atlas.types import Plan
from atlas.types import StepResult


class Database:
    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None

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

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_session(self, task: str, metadata: Dict[str, Any] | None = None) -> int:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            return await connection.fetchval(
                "INSERT INTO sessions(task, metadata) VALUES ($1, $2) RETURNING id",
                task,
                metadata,
            )

    async def log_plan(self, session_id: int, plan: Plan) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO plans(session_id, plan) VALUES ($1, $2)"
                " ON CONFLICT (session_id) DO UPDATE SET plan = EXCLUDED.plan",
                session_id,
                plan.model_dump(),
            )

    async def log_step_result(self, session_id: int, result: StepResult) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO step_results(session_id, step_id, trace, output, evaluation, attempts)"
                " VALUES ($1, $2, $3, $4, $5, $6)"
                " ON CONFLICT (session_id, step_id) DO UPDATE SET"
                " trace = EXCLUDED.trace, output = EXCLUDED.output, evaluation = EXCLUDED.evaluation, attempts = EXCLUDED.attempts",
                session_id,
                result.step_id,
                result.trace,
                result.output,
                result.evaluation,
                result.attempts,
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
                (session_id, step_id, attempt.get("attempt", index + 1), attempt.get("evaluation"))
                for index, attempt in enumerate(attempts)
            ]
            if records:
                await connection.executemany(
                    "INSERT INTO step_attempts(session_id, step_id, attempt, evaluation) VALUES ($1, $2, $3, $4)",
                    records,
                )

    async def log_intermediate_step(self, session_id: int, event: IntermediateStep) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO trajectory_events(session_id, event) VALUES ($1, $2)",
                session_id,
                event.model_dump(),
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

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database connection has not been established")
        return self._pool
