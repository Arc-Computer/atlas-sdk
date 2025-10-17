"""Atlas CLI entry point supporting triage scaffolding and optional storage helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent, indent

from atlas.cli.storage_runtime import InitOptions, QuitOptions, init_storage, quit_storage


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
    # Example metadata enrichment:
    # builder.update_metadata(notes="Why this request matters.")
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


def _cmd_init_storage(args: argparse.Namespace) -> int:
    options = InitOptions(
        compose_file=Path(args.compose_file),
        force=args.force,
        no_start=args.no_start,
        auto_install=not args.skip_docker_install,
    )
    return init_storage(options)


def _cmd_quit_storage(args: argparse.Namespace) -> int:
    options = QuitOptions(
        compose_file=Path(args.compose_file),
        purge=args.purge,
    )
    return quit_storage(options)


def _cmd_storage_up(args: argparse.Namespace) -> int:
    print("`atlas storage up` is deprecated. Use `atlas init` instead.\n")
    return _cmd_init_storage(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas SDK command-line tools for triage scaffolding and storage provisioning.",
    )
    parser.set_defaults(handler=lambda args: parser.print_help() or 0)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    triage_parser = subparsers.add_parser("triage", help="Triage helper commands.")
    triage_parser.set_defaults(handler=lambda args: triage_parser.print_help() or 0)
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

    init_storage_parser = subparsers.add_parser(
        "init",
        help="Provision local storage (Docker, Postgres, and schema).",
    )
    init_storage_parser.add_argument(
        "--compose-file",
        default="atlas-postgres.yaml",
        help="Path where the compose file will be written (default: %(default)s).",
    )
    init_storage_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the compose file if it already exists.",
    )
    init_storage_parser.add_argument(
        "--skip-docker-install",
        action="store_true",
        help="Assume Docker is already installed and skip automatic installation.",
    )
    init_storage_parser.add_argument(
        "--no-start",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    init_storage_parser.set_defaults(handler=_cmd_init_storage)

    quit_parser = subparsers.add_parser(
        "quit",
        help="Stop Atlas storage services and optionally remove volumes.",
    )
    quit_parser.add_argument(
        "--compose-file",
        default="atlas-postgres.yaml",
        help="Compose file created by `atlas init` (default: %(default)s).",
    )
    quit_parser.add_argument(
        "--purge",
        action="store_true",
        help="Remove Docker volumes in addition to stopping containers.",
    )
    quit_parser.set_defaults(handler=_cmd_quit_storage)

    storage_parser = subparsers.add_parser("storage", help=argparse.SUPPRESS)
    storage_parser.set_defaults(handler=lambda args: storage_parser.print_help() or 0)
    storage_subparsers = storage_parser.add_subparsers(dest="storage_command", metavar="<subcommand>")

    up_parser = storage_subparsers.add_parser(
        "up",
        help=argparse.SUPPRESS,
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
    up_parser.add_argument(
        "--skip-docker-install",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    up_parser.set_defaults(handler=_cmd_storage_up)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
