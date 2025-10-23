"""Asynchronous PostgreSQL persistence layer."""

from __future__ import annotations

import json
from statistics import fmean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from importlib import resources as importlib_resources
except ImportError:  # pragma: no cover
    import importlib_resources  # type: ignore

try:
    import asyncpg  # type: ignore[import-untyped]
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
        evaluation_payload: Any
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

    async def log_discovery_run(
        self,
        *,
        project_root: str,
        task: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
        source: str = "discovery",
    ) -> int:
        pool = self._require_pool()
        serialized_payload = self._serialize_json(payload) or "{}"
        serialized_metadata = self._serialize_json(metadata) if metadata else None
        async with pool.acquire() as connection:
            return await connection.fetchval(
                "INSERT INTO discovery_runs(project_root, task, source, payload, metadata)"
                " VALUES ($1, $2, $3, $4, $5) RETURNING id",
                project_root,
                task,
                source,
                serialized_payload,
                serialized_metadata,
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
        reward_stats: Optional[Dict[str, Any]] = None,
        reward_audit: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        pool = self._require_pool()
        serialized_reward = self._serialize_json(reward.to_dict() if hasattr(reward, "to_dict") else reward) if reward else None
        serialized_stats = self._serialize_json(reward_stats) if reward_stats else None
        serialized_audit = self._serialize_json(list(reward_audit)) if reward_audit else None
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE sessions SET reward = $1, student_learning = $2, teacher_learning = $3, reward_stats = $4, reward_audit = $5 WHERE id = $6",
                serialized_reward,
                student_learning,
                teacher_learning,
                serialized_stats,
                serialized_audit,
                session_id,
            )

    async def fetch_sessions(self, limit: int = 50, offset: int = 0) -> List[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, task, status, review_status, review_notes, metadata, final_answer, reward, reward_stats, reward_audit, student_learning, teacher_learning, created_at, completed_at"
                " FROM sessions ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
        return [dict(row) for row in rows]

    async def fetch_session(self, session_id: int) -> dict[str, Any] | None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT id, task, status, review_status, review_notes, metadata, final_answer, reward, reward_stats, reward_audit, student_learning, teacher_learning, created_at, completed_at"
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

    async def list_sessions_by_status(self, review_status: str, limit: int = 50, offset: int = 0) -> List[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, task, status, review_status, review_notes, metadata, final_answer, reward, reward_stats, reward_audit, student_learning, teacher_learning, created_at, completed_at"
                " FROM sessions WHERE review_status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                review_status,
                limit,
                offset,
            )
        return [dict(row) for row in rows]

    async def update_session_review_status(
        self,
        session_id: int,
        review_status: str,
        notes: Optional[str] = None,
    ) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            if notes is None:
                await connection.execute(
                    "UPDATE sessions SET review_status = $1 WHERE id = $2",
                    review_status,
                    session_id,
                )
            else:
                await connection.execute(
                    "UPDATE sessions SET review_status = $1, review_notes = $2 WHERE id = $3",
                    review_status,
                    notes,
                    session_id,
                )

    async def fetch_reward_baseline(
        self,
        learning_key: Optional[str] = None,
        *,
        window: int = 50,
    ) -> dict[str, Any]:
        pool = self._require_pool()
        constraints: list[str] = ["reward_stats IS NOT NULL"]
        params: list[Any] = []
        if learning_key:
            constraints.append("(metadata ->> 'learning_key') = $" + str(len(params) + 1))
            params.append(learning_key)
        limit_index = len(params) + 1
        params.append(max(window, 1))
        where_clause = " AND ".join(constraints)
        query = (
            "SELECT reward_stats FROM sessions"
            f" WHERE {where_clause}"
            " ORDER BY created_at DESC LIMIT $" + str(limit_index)
        )
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
        stats_payloads: list[dict[str, Any]] = []
        for row in rows:
            raw_stats = row.get("reward_stats")
            payload = self._deserialize_json(raw_stats)
            if isinstance(payload, dict):
                stats_payloads.append(payload)
        return self._aggregate_reward_baseline(stats_payloads, window=window)

    async def fetch_learning_keys(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        project_root: str | None = None,
        task: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> List[dict[str, Any]]:
        pool = self._require_pool()
        params: list[Any] = []
        constraints: list[str] = ["metadata->>'learning_key' IS NOT NULL"]
        if project_root:
            params.append(project_root)
            constraints.append(f"(metadata ->> 'project_root') = ${len(params)}")
        if task:
            params.append(task)
            constraints.append(f"task = ${len(params)}")
        if tags:
            filtered_tags = [tag for tag in tags if tag]
            for tag_value in filtered_tags:
                params.append(tag_value)
                constraints.append(
                    f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'tags') AS tag WHERE tag = ${len(params)})"
                )
        query = (
            "SELECT metadata->>'learning_key' AS learning_key,"
            " COUNT(*) AS session_count,"
            " MIN(created_at) AS first_seen,"
            " MAX(created_at) AS last_seen"
            " FROM sessions"
        )
        if constraints:
            query += " WHERE " + " AND ".join(constraints)
        query += (
            " GROUP BY learning_key"
            " ORDER BY session_count DESC, last_seen DESC"
        )
        if limit is not None:
            params.append(max(int(limit), 0))
            query += f" LIMIT ${len(params)}"
        if offset:
            params.append(max(int(offset), 0))
            query += f" OFFSET ${len(params)}"
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_sessions_for_learning_key(
        self,
        learning_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        if not learning_key:
            return []
        pool = self._require_pool()
        params: list[Any] = [learning_key]
        query = (
            "SELECT id, task, status, review_status, metadata, final_answer,"
            " reward, reward_stats, reward_audit, student_learning, teacher_learning,"
            " created_at, completed_at"
            " FROM sessions"
            " WHERE metadata->>'learning_key' = $1"
            " ORDER BY created_at ASC"
        )
        if limit is not None:
            params.append(max(int(limit), 0))
            query += f" LIMIT ${len(params)}"
        if offset:
            params.append(max(int(offset), 0))
            query += f" OFFSET ${len(params)}"
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_learning_sessions(
        self,
        *,
        learning_key: str | None = None,
        project_root: str | None = None,
        task: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        order: str = "asc",
    ) -> List[dict[str, Any]]:
        pool = self._require_pool()
        params: list[Any] = []
        clauses: list[str] = []
        if learning_key:
            params.append(learning_key)
            clauses.append(f"(metadata ->> 'learning_key') = ${len(params)}")
        if project_root:
            params.append(project_root)
            clauses.append(f"(metadata ->> 'project_root') = ${len(params)}")
        if task:
            params.append(task)
            clauses.append(f"task = ${len(params)}")
        if tags:
            filtered_tags = [tag for tag in tags if tag]
            for tag_value in filtered_tags:
                params.append(tag_value)
                clauses.append(
                    f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'tags') AS tag WHERE tag = ${len(params)})"
                )
        ordering = "ASC" if str(order).lower() != "desc" else "DESC"
        query = (
            "SELECT id, task, status, review_status, metadata, reward, reward_stats, reward_audit,"
            " student_learning, teacher_learning, created_at, completed_at,"
            " metadata->>'learning_key' AS learning_key"
            " FROM sessions"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += f" ORDER BY created_at {ordering}"
        if limit is not None:
            params.append(max(int(limit), 0))
            query += f" LIMIT ${len(params)}"
        if offset:
            params.append(max(int(offset), 0))
            query += f" OFFSET ${len(params)}"
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_discovery_runs(
        self,
        *,
        project_root: str | None = None,
        task: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        pool = self._require_pool()
        params: list[Any] = []
        clauses: list[str] = []
        if project_root:
            params.append(project_root)
            clauses.append(f"project_root = ${len(params)}")
        if task:
            params.append(task)
            clauses.append(f"task = ${len(params)}")
        if source:
            sources = list(source) if isinstance(source, (list, tuple, set)) else [source]
            sources = [item for item in sources if item is not None]
            if sources:
                placeholders: list[str] = []
                for value in sources:
                    params.append(value)
                    placeholders.append(f"${len(params)}")
                clauses.append(f"source IN ({', '.join(placeholders)})")
        query = "SELECT id, project_root, task, source, payload, metadata, created_at FROM discovery_runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            params.append(max(int(limit), 0))
            query += f" LIMIT ${len(params)}"
        if offset:
            params.append(max(int(offset), 0))
            query += f" OFFSET ${len(params)}"
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_trajectory_event_counts(self, session_ids: Sequence[int]) -> Dict[int, int]:
        if not session_ids:
            return {}
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT session_id, COUNT(*) AS event_count"
                " FROM trajectory_events"
                " WHERE session_id = ANY($1::int[])"
                " GROUP BY session_id",
                session_ids,
            )
        return {int(row["session_id"]): int(row["event_count"]) for row in rows}

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

    async def fetch_learning_state(self, learning_key: str) -> dict[str, Any] | None:
        if not learning_key:
            return None
        pool = self._require_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT student_learning, teacher_learning, metadata, updated_at"
                " FROM learning_registry WHERE learning_key = $1",
                learning_key,
            )
        if row is None:
            return None
        metadata = self._deserialize_json(row.get("metadata"))
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "student_learning": row.get("student_learning") or "",
            "teacher_learning": row.get("teacher_learning"),
            "metadata": metadata,
            "updated_at": row.get("updated_at"),
        }

    async def upsert_learning_state(
        self,
        learning_key: str,
        student_learning: Optional[str],
        teacher_learning: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not learning_key:
            return
        pool = self._require_pool()
        serialized_metadata = self._serialize_json(metadata) if metadata else None
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO learning_registry(learning_key, student_learning, teacher_learning, metadata, updated_at)"
                " VALUES ($1, $2, $3, $4, NOW())"
                " ON CONFLICT (learning_key) DO UPDATE SET"
                " student_learning = EXCLUDED.student_learning,"
                " teacher_learning = EXCLUDED.teacher_learning,"
                " metadata = EXCLUDED.metadata,"
                " updated_at = NOW()",
                learning_key,
                student_learning,
                teacher_learning,
                serialized_metadata,
            )

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

    @staticmethod
    def _aggregate_reward_baseline(
        entries: Iterable[Dict[str, Any]],
        *,
        window: int,
    ) -> dict[str, Any]:
        snapshots = list(entries)
        sample_count = len(snapshots)
        score_values: list[float] = []
        uncertainty_values: list[float] = []
        best_uncertainties: list[float] = []
        for snapshot in snapshots:
            score_value = Database._coerce_float(
                snapshot.get("score_mean", snapshot.get("score"))
            )
            if score_value is not None:
                score_values.append(score_value)
            uncertainty_mean = Database._coerce_float(snapshot.get("uncertainty_mean"))
            if uncertainty_mean is not None:
                uncertainty_values.append(uncertainty_mean)
            best_uncertainty = Database._coerce_float(
                snapshot.get("best_uncertainty", snapshot.get("min_uncertainty"))
            )
            if best_uncertainty is not None:
                best_uncertainties.append(best_uncertainty)
        baseline: dict[str, Any] = {
            "window": window,
            "sample_count": sample_count,
            "score_mean": fmean(score_values) if score_values else None,
            "score_median": median(score_values) if score_values else None,
            "score_stddev": pstdev(score_values) if len(score_values) > 1 else (0.0 if score_values else None),
            "uncertainty_mean": fmean(uncertainty_values) if uncertainty_values else None,
            "uncertainty_median": median(uncertainty_values) if uncertainty_values else None,
            "uncertainty_stddev": pstdev(uncertainty_values) if len(uncertainty_values) > 1 else (0.0 if uncertainty_values else None),
            "best_uncertainty_mean": fmean(best_uncertainties) if best_uncertainties else None,
            "best_uncertainty_median": median(best_uncertainties) if best_uncertainties else None,
            "best_uncertainty_stddev": pstdev(best_uncertainties) if len(best_uncertainties) > 1 else (0.0 if best_uncertainties else None),
            "scores": score_values,
            "uncertainties": uncertainty_values,
            "best_uncertainties": best_uncertainties,
        }
        return baseline

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

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
