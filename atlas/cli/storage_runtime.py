"""Helper routines for `atlas init` / `atlas quit`.

This module intentionally keeps a very small surface area.  The CLI wraps it to
provision a local PostgreSQL instance (via Docker Compose) and apply the Atlas
schema so first–time users can persist sessions without writing bespoke infra.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database

DEFAULT_DATABASE_URL = "postgresql://atlas:atlas@localhost:5433/atlas"
DEFAULT_MIN_CONNECTIONS = 1
DEFAULT_MAX_CONNECTIONS = 5
DEFAULT_STATEMENT_TIMEOUT = 30.0

_COMPOSE_TEMPLATE = textwrap.dedent(
    """
    services:
      postgres:
        container_name: atlas-postgres
        image: postgres:16
        restart: unless-stopped
        environment:
          POSTGRES_DB: atlas
          POSTGRES_USER: atlas
          POSTGRES_PASSWORD: atlas
        volumes:
          - atlas_pg_data:/var/lib/postgresql/data
        ports:
          - "5433:5432"

    volumes:
      atlas_pg_data:
        driver: local
    """
).strip() + "\n"


class StorageRuntimeError(RuntimeError):
    """Wrapper exception so the CLI can surface friendly messages."""


@dataclass(slots=True)
class InitOptions:
    compose_file: Path
    force: bool = False
    no_start: bool = False
    auto_install: bool = True
    database_url: str = DEFAULT_DATABASE_URL
    status_printer: Callable[[str], None] = print


@dataclass(slots=True)
class QuitOptions:
    compose_file: Path
    purge: bool = False
    status_printer: Callable[[str], None] = print


def init_storage(options: InitOptions) -> int:
    """Writable entry point for `atlas init`."""

    try:
        _ensure_docker(options)
        _write_compose(options)
        if not options.no_start:
            _start_services(options)
            _apply_schema(options)
    except StorageRuntimeError as exc:
        options.status_printer(str(exc))
        return 1
    return 0


def quit_storage(options: QuitOptions) -> int:
    """Entry point for `atlas quit`."""

    compose_file = options.compose_file
    if not compose_file.exists():
        options.status_printer(f"Compose file {compose_file} not found; nothing to stop.")
        return 0

    args = ["docker", "compose", "-f", str(compose_file), "down"]
    if options.purge:
        args.append("--volumes")
    return _run_command(args, options.status_printer)


def _ensure_docker(options: InitOptions) -> None:
    if shutil.which("docker"):
        return
    if not options.auto_install:
        raise StorageRuntimeError(
            "Docker is required but was not found on PATH. Install Docker Desktop, or rerun `atlas init` "
            "without --skip-docker-install once Docker is available."
        )
    system = platform.system()
    if system == "Darwin":
        command = ["brew", "install", "--cask", "docker"]
    elif system == "Linux":
        command = [
            "sh",
            "-c",
            "curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh",
        ]
    else:
        raise StorageRuntimeError(
            "Automatic Docker installation is unavailable on this platform. Please install Docker manually."
        )
    options.status_printer("Docker not detected. Attempting installation...")
    exit_code = _run_command(command, options.status_printer)
    if exit_code != 0:
        raise StorageRuntimeError(
            "Docker installation failed. Install Docker manually and rerun `atlas init`."
        )


def _write_compose(options: InitOptions) -> None:
    compose_file = options.compose_file
    if compose_file.exists() and not options.force:
        raise StorageRuntimeError(
            f"{compose_file} already exists. Pass --force to overwrite."
        )
    compose_file.parent.mkdir(parents=True, exist_ok=True)
    compose_file.write_text(_COMPOSE_TEMPLATE, encoding="utf-8")
    options.status_printer(f"Wrote Docker compose file to {compose_file}")


def _start_services(options: InitOptions) -> None:
    args = ["docker", "compose", "-f", str(options.compose_file), "up", "-d", "postgres"]
    options.status_printer("Starting PostgreSQL container...")
    exit_code = _run_command(args, options.status_printer)
    if exit_code != 0:
        raise StorageRuntimeError(
            "Failed to start postgres via Docker Compose. See the logs above for details."
        )


def _apply_schema(options: InitOptions) -> None:
    async def _ensure_schema() -> None:
        config = StorageConfig(
            database_url=options.database_url,
            min_connections=DEFAULT_MIN_CONNECTIONS,
            max_connections=DEFAULT_MAX_CONNECTIONS,
            statement_timeout_seconds=DEFAULT_STATEMENT_TIMEOUT,
        )
        database = Database(config)
        last_error: Exception | None = None
        for attempt in range(1, 31):
            try:
                options.status_printer(f"Waiting for database (attempt {attempt}/30)...")
                await database.connect()
                await database.disconnect()
                options.status_printer("Database ready; schema ensured.")
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(1)
        raise StorageRuntimeError(
            f"Could not connect to database at {options.database_url}. "
            f"Last error: {last_error}"
        )

    asyncio.run(_ensure_schema())


def _run_command(args: list[str], printer: Callable[[str], None]) -> int:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        printer(proc.stdout.strip())
    if proc.returncode != 0 and proc.returncode != 130:
        printer(f"Command {' '.join(args)} exited with status {proc.returncode}")
    return proc.returncode
