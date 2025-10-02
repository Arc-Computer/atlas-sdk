import asyncio
import os
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

pytest.importorskip("fastapi")
asyncpg = pytest.importorskip("asyncpg")

from atlas.dashboard.server import create_dashboard_app


@pytest.mark.postgres
def test_dashboard_endpoints_with_postgres():
    async def runner():
        database_url = os.getenv("ATLAS_TEST_DATABASE_URL")
        if not database_url:
            pytest.skip("ATLAS_TEST_DATABASE_URL not set")

        schema_sql = Path("atlas/storage/schema.sql").read_text()
        conn = await asyncpg.connect(dsn=database_url)
        task_label = f"dashboard-smoke-{uuid.uuid4()}"
        try:
            await conn.execute(schema_sql)
            session_id = await conn.fetchval(
                "INSERT INTO sessions(task, status, final_answer) VALUES($1, $2, $3) RETURNING id",
                task_label,
                "succeeded",
                "completed",
            )
            await conn.execute(
                "INSERT INTO plans(session_id, plan) VALUES($1, $2)"
                " ON CONFLICT (session_id) DO UPDATE SET plan = EXCLUDED.plan",
                session_id,
                {"steps": []},
            )
            await conn.execute(
                "INSERT INTO step_results(session_id, step_id, trace, output, evaluation, attempts)"
                " VALUES($1, $2, $3, $4, $5, $6)"
                " ON CONFLICT (session_id, step_id) DO UPDATE SET output = EXCLUDED.output",
                session_id,
                1,
                "trace",
                "output",
                {"score": 1.0},
                1,
            )
            await conn.execute(
                "INSERT INTO trajectory_events(session_id, event) VALUES($1, $2)",
                session_id,
                {"payload": {"name": "smoke"}},
            )
        finally:
            await conn.close()

        app = create_dashboard_app(database_url=database_url)
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/sessions")
            assert response.status_code == 200
            sessions = response.json()["sessions"]
            assert any(session["task"] == task_label for session in sessions)

        conn = await asyncpg.connect(dsn=database_url)
        try:
            await conn.execute("DELETE FROM sessions WHERE task = $1", task_label)
        finally:
            await conn.close()

    asyncio.run(runner())
