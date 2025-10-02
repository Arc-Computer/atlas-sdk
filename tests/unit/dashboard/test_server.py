import asyncio

import pytest
from httpx import ASGITransport
from httpx import AsyncClient

pytest.importorskip("fastapi")

from atlas.dashboard.publisher import TelemetryPublisher
from atlas.dashboard.server import create_dashboard_app


class FakeRepository:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self._session = {
            "id": 7,
            "task": "demo",
            "status": "succeeded",
            "final_answer": "answer",
            "created_at": "2025-01-01T00:00:00Z",
            "completed_at": "2025-01-01T00:00:01Z",
            "plan": {"steps": []},
            "metadata": {},
        }
        self._sessions = [self._session]
        self._steps = [
            {
                "step_id": 1,
                "trace": "trace",
                "output": "output",
                "evaluation": {"score": 1.0},
                "attempts": 1,
                "attempt_details": [],
                "guidance_notes": [],
            }
        ]
        self._events = [
            {"id": 1, "created_at": "2025-01-01T00:00:00Z", "event": {"payload": {"name": "demo"}}}
        ]

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def list_sessions(self, limit: int = 50, offset: int = 0):
        return self._sessions[offset : offset + limit]

    async def get_session(self, session_id: int):
        if session_id == self._session["id"]:
            return self._session
        return None

    async def get_session_steps(self, session_id: int):
        if session_id != self._session["id"]:
            return []
        return self._steps

    async def get_session_events(self, session_id: int, limit: int = 200):
        if session_id != self._session["id"]:
            return []
        return self._events[:limit]


def test_list_sessions_returns_repository_data():
    async def runner():
        repository = FakeRepository()
        app = create_dashboard_app(repository=repository, publisher=TelemetryPublisher())
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json()["sessions"][0]["id"] == 7
        assert repository.connected is True
        assert repository.disconnected is True

    asyncio.run(runner())


def test_session_detail_includes_plan():
    async def runner():
        repository = FakeRepository()
        app = create_dashboard_app(repository=repository, publisher=TelemetryPublisher())
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/api/sessions/7")
        assert response.status_code == 200
        body = response.json()
        assert body["session"]["plan"] == {"steps": []}

    asyncio.run(runner())


def test_steps_endpoint_returns_data():
    async def runner():
        repository = FakeRepository()
        app = create_dashboard_app(repository=repository, publisher=TelemetryPublisher())
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/api/sessions/7/steps")
        assert response.status_code == 200
        steps = response.json()["steps"]
        assert steps[0]["evaluation"]["score"] == 1.0

    asyncio.run(runner())


def test_session_not_found_returns_404():
    async def runner():
        repository = FakeRepository()
        app = create_dashboard_app(repository=repository, publisher=TelemetryPublisher())
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/api/sessions/999")
        assert response.status_code == 404

    asyncio.run(runner())
