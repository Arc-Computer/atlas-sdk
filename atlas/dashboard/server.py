"""FastAPI service exposing Atlas telemetry and storage visibility."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import HTMLResponse

from atlas.config.models import StorageConfig
from atlas.dashboard.publisher import TelemetryPublisher
from atlas.dashboard.repository import DashboardRepository
from atlas.dashboard.templates import DASHBOARD_HTML
from atlas.storage.database import Database

router = APIRouter(prefix="/api")


async def get_repository(request: Request) -> DashboardRepository:
    repository: DashboardRepository | None = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Repository not configured")
    return repository


def create_dashboard_app(
    *,
    database_url: str | None = None,
    repository: DashboardRepository | None = None,
    publisher: TelemetryPublisher | None = None,
    min_connections: int = 1,
    max_connections: int = 5,
    statement_timeout_seconds: float = 30.0,
) -> FastAPI:
    if repository is None:
        if database_url is None:
            raise ValueError("database_url or repository must be provided")
        storage_config = StorageConfig(
            database_url=database_url,
            min_connections=min_connections,
            max_connections=max_connections,
            statement_timeout_seconds=statement_timeout_seconds,
        )
        repository = DashboardRepository(Database(storage_config))
    publisher = publisher or TelemetryPublisher()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await repository.connect()
        try:
            yield
        finally:
            await repository.disconnect()

    app = FastAPI(title="Atlas Telemetry Dashboard", lifespan=lifespan)
    app.state.repository = repository
    app.state.publisher = publisher

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_root() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        await websocket.accept()
        stream = publisher.stream()
        try:
            async for payload in stream:
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await stream.aclose()

    return app


@router.get("/sessions")
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    repository: DashboardRepository = Depends(get_repository),
) -> dict[str, Any]:
    sessions = await repository.list_sessions(limit=limit, offset=offset)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: int,
    repository: DashboardRepository = Depends(get_repository),
) -> dict[str, Any]:
    session = await repository.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@router.get("/sessions/{session_id}/steps")
async def get_session_steps(
    session_id: int,
    repository: DashboardRepository = Depends(get_repository),
) -> dict[str, Any]:
    steps = await repository.get_session_steps(session_id)
    return {"steps": steps}


@router.get("/sessions/{session_id}/events")
async def get_session_events(
    session_id: int,
    limit: int = 200,
    repository: DashboardRepository = Depends(get_repository),
) -> dict[str, Any]:
    events = await repository.get_session_events(session_id, limit=limit)
    return {"events": events}


def run_dashboard(
    *,
    database_url: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    publisher: TelemetryPublisher | None = None,
) -> None:
    app = create_dashboard_app(database_url=database_url, publisher=publisher)
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas telemetry dashboard server")
    parser.add_argument("--database-url", required=True, help="PostgreSQL database URL")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (debug)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dashboard(
        database_url=args.database_url,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
