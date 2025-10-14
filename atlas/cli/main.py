"""Atlas CLI entry point supporting triage scaffolding and optional storage helpers."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent, indent


def _format_snippet(snippet: str) -> str:
    """Indent a snippet so it lands inside the generated function body."""

    cleaned = dedent(snippet).strip("\n")
    if not cleaned:
        return ""
    return indent(cleaned, "    ") + "\n"


_DOMAIN_SNIPPETS = {
    "sre": """\
builder.set_summary("Investigate production incident and restore service availability.")
builder.add_tags("domain:sre")
builder.add_risk("Potential customer impact if MTTR breaches SLA.", severity="high")
builder.add_signal("alert.count", metadata.get("alert_count", 0))
""",
    "support": """\
builder.set_summary("Customer support follow-up to unblock the account.")
builder.add_tags("domain:support")
builder.add_risk("Negative customer sentiment escalation.", severity="moderate")
builder.add_signal("customer.sentiment", metadata.get("sentiment", "neutral"))
""",
    "code": """\
builder.set_summary("Debug failing tests and ship a fix.")
builder.add_tags("domain:code")
builder.add_risk("CI deployment blocked until failures resolved.", severity="high")
builder.add_signal("ci.failing_tests", metadata.get("failing_tests", []))
""",
}


_BASE_TEMPLATE = """from __future__ import annotations

from typing import Any, Dict

from atlas.utils.triage import TriageDossier, TriageDossierBuilder

# Tip: see examples.triage_adapters for more opinionated recipes.


def {function_name}(task: str, metadata: Dict[str, Any] | None = None) -> TriageDossier:
    metadata = metadata or {{}}
    builder = TriageDossierBuilder(task=task)
{domain_snippet}
    # Example persona reference:
    # builder.add_persona_reference("persona-id", rationale="Why it's relevant.", weight=1.0)
    return builder.build()
"""


def _write_template(path: Path, template: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")


def _cmd_triage_init(args: argparse.Namespace) -> int:
    domain = (args.domain or "custom").lower()
    raw_snippet = _DOMAIN_SNIPPETS.get(domain)
    if raw_snippet is None:
        lines = [
            'builder.set_summary("Describe the task you are triaging.")',
            'builder.add_tags("domain:custom")',
        ]
        snippet = _format_snippet("\n".join(lines))
    else:
        snippet = _format_snippet(raw_snippet)
    template = _BASE_TEMPLATE.format(function_name=args.function_name, domain_snippet=snippet)
    try:
        _write_template(Path(args.output), template, force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Created triage adapter scaffold at {args.output}")
    return 0


_COMPOSE_TEMPLATE = """version: "3.9"

services:
  postgres:
    image: postgres:15
    container_name: atlas-postgres
    environment:
      POSTGRES_USER: atlas
      POSTGRES_PASSWORD: atlas
      POSTGRES_DB: atlas
    ports:
      - "5433:5432"
    volumes:
      - atlas_pg_data:/var/lib/postgresql/data

volumes:
  atlas_pg_data:
"""


def _compose_command() -> list[str] | None:
    docker = shutil.which("docker")
    if docker is not None:
        return [docker, "compose"]
    docker_compose = shutil.which("docker-compose")
    if docker_compose is not None:
        return [docker_compose]
    return None


def _cmd_storage_up(args: argparse.Namespace) -> int:
    compose_path = Path(args.compose_file).expanduser().resolve()
    compose_path.parent.mkdir(parents=True, exist_ok=True)
    if compose_path.exists() and not args.force:
        print(f"{compose_path} already exists; use --force to overwrite.", file=sys.stderr)
        return 1
    compose_path.write_text(_COMPOSE_TEMPLATE, encoding="utf-8")
    print(f"Wrote Docker Compose file to {compose_path}")
    print("Storage is optional—skip this step when you only need in-memory runs.")

    if args.no_start:
        _print_storage_instructions(compose_path)
        return 0

    command = _compose_command()
    if command is None:
        print(
            "Docker is not available on PATH. Install Docker and run the following command manually:\n"
            f"  docker compose -f {compose_path} up -d postgres",
            file=sys.stderr,
        )
        _print_storage_instructions(compose_path)
        return 1

    cmd = command + ["-f", str(compose_path), "up", "-d", "postgres"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to start Postgres via Docker: {exc}", file=sys.stderr)
        _print_storage_instructions(compose_path)
        return 1

    print("Postgres is running in Docker.")
    _print_storage_instructions(compose_path)
    return 0


def _print_storage_instructions(compose_path: Path) -> None:
    connection_url = "postgresql://atlas:atlas@localhost:5433/atlas"
    print()
    print("To connect Atlas to this instance (optional), set either of the following:")
    print(f"  export STORAGE__DATABASE_URL={connection_url}")
    print("or add the same value to your Atlas YAML config under storage.database_url.")
    print()
    print("To stop the container, run:")
    print(f"  docker compose -f {compose_path} down")
    print()
    print("Once connected, CLI exports will include adaptive summaries, reward highlights, and persona usage telemetry.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas SDK command-line tools for triage scaffolding and optional storage utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    triage_parser = subparsers.add_parser("triage", help="Triage helper commands.")
    triage_subparsers = triage_parser.add_subparsers(dest="triage_command", metavar="<subcommand>")

    init_parser = triage_subparsers.add_parser("init", help="Generate a triage adapter scaffold.")
    init_parser.add_argument("--output", default="triage_adapter.py", help="Destination path for the generated adapter.")
    init_parser.add_argument(
        "--domain",
        choices=["sre", "support", "code", "custom"],
        default="custom",
        help="Domain template to pre-populate signals and risks.",
    )
    init_parser.add_argument(
        "--function-name",
        default="build_dossier",
        help="Name of the factory function exported by the adapter.",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
    init_parser.set_defaults(handler=_cmd_triage_init)

    storage_parser = subparsers.add_parser("storage", help="Optional storage helpers for PostgreSQL persistence.")
    storage_subparsers = storage_parser.add_subparsers(dest="storage_command", metavar="<subcommand>")

    up_parser = storage_subparsers.add_parser(
        "up",
        help="Write a Docker Compose file for Postgres and optionally start it.",
    )
    up_parser.add_argument(
        "--compose-file",
        default="atlas-postgres.yaml",
        help="Path where the compose file will be written (default: %(default)s).",
    )
    up_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the compose file if it already exists.",
    )
    up_parser.add_argument(
        "--no-start",
        action="store_true",
        help="Only write the compose file without starting Docker.",
    )
    up_parser.set_defaults(handler=_cmd_storage_up)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if args.command == "triage":
        if not getattr(args, "triage_command", None):
            parser.parse_args(["triage", "--help"])
            return 0
        handler = getattr(args, "handler", None)
        if handler is None:
            parser.parse_args(["triage", args.triage_command, "--help"])
            return 0
        return handler(args)
    if args.command == "storage":
        if not getattr(args, "storage_command", None):
            parser.parse_args(["storage", "--help"])
            return 0
        handler = getattr(args, "handler", None)
        if handler is None:
            parser.parse_args(["storage", args.storage_command, "--help"])
            return 0
        return handler(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
