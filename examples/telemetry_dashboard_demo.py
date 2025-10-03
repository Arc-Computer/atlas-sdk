"""Launch the Atlas telemetry dashboard alongside a sample run."""

from __future__ import annotations

import argparse
import asyncio
import contextlib

import uvicorn

from atlas.core import arun
from atlas.dashboard import TelemetryPublisher
from atlas.dashboard import create_dashboard_app


class BackgroundServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # pragma: no cover
        return


async def start_dashboard_server(app, host: str, port: int) -> tuple[BackgroundServer, asyncio.Task[None]]:
    config = uvicorn.Config(app=app, host=host, port=port, loop="asyncio", log_level="info")
    server = BackgroundServer(config)
    task = asyncio.create_task(server.serve())
    while not server.started.is_set():
        await asyncio.sleep(0.05)
    return server, task


async def stream_to_console(publisher: TelemetryPublisher) -> None:
    async for payload in publisher.stream():
        data = payload.get("data", {})
        kind = payload.get("type", "event")
        name = data.get("payload", {}).get("name") if isinstance(data, dict) else None
        event_type = data.get("payload", {}).get("event_type") if isinstance(data, dict) else None
        parts = [kind]
        if event_type:
            parts.append(str(event_type))
        if name:
            parts.append(str(name))
        print("[telemetry]", " · ".join(parts))


async def run_with_dashboard(args) -> None:
    publisher = TelemetryPublisher()
    app = create_dashboard_app(database_url=args.database_url, publisher=publisher)
    server, server_task = await start_dashboard_server(app, args.host, args.port)
    console_task = asyncio.create_task(stream_to_console(publisher))
    try:
        result = await arun(args.task, args.config, publisher=publisher)
        print("Final answer:", result.final_answer)
    finally:
        console_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await console_task
        server.should_exit = True
        await server_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Atlas with the telemetry dashboard enabled")
    parser.add_argument("--task", default="Summarize the Atlas SDK repository.", help="Natural language task to execute")
    parser.add_argument(
        "--config",
        default="configs/examples/python_agent.yaml",
        help="Path to the Atlas configuration YAML",
    )
    parser.add_argument("--database-url", required=True, help="PostgreSQL connection URL used by Atlas persistence")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host bind address")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_with_dashboard(args))


if __name__ == "__main__":
    main()
