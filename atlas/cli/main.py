"""Atlas CLI entry point supporting triage scaffolding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent


_DOMAIN_SNIPPETS = {
    "sre": dedent(
        """\
        builder.set_summary("Investigate production incident and restore service availability.")
        builder.add_tags("domain:sre")
        builder.add_risk("Potential customer impact if MTTR breaches SLA.", severity="high")
        builder.add_signal("alert.count", metadata.get("alert_count", 0))
        """
    ),
    "support": dedent(
        """\
        builder.set_summary("Customer support follow-up to unblock the account.")
        builder.add_tags("domain:support")
        builder.add_risk("Negative customer sentiment escalation.", severity="moderate")
        builder.add_signal("customer.sentiment", metadata.get("sentiment", "neutral"))
        """
    ),
    "code": dedent(
        """\
        builder.set_summary("Debug failing tests and ship a fix.")
        builder.add_tags("domain:code")
        builder.add_risk("CI deployment blocked until failures resolved.", severity="high")
        builder.add_signal("ci.failing_tests", metadata.get("failing_tests", []))
        """
    ),
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
    snippet = _DOMAIN_SNIPPETS.get(domain, "    builder.set_summary(\"Describe the task you are triaging.\")\n")
    if domain not in _DOMAIN_SNIPPETS:
        snippet += '    builder.add_tags("domain:custom")\n'
    template = _BASE_TEMPLATE.format(function_name=args.function_name, domain_snippet=snippet)
    try:
        _write_template(Path(args.output), template, force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Created triage adapter scaffold at {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="Atlas SDK command-line tools.")
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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
