"""CLI helpers for exporting Atlas traces and triggering Atlas Core training."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

from atlas.cli.export import add_export_arguments, configure_logging, resolve_review_filters
from atlas.cli.jsonl_writer import ExportRequest, export_sessions_sync

_TRAIN_HELP = "Export runtime traces and launch the Atlas Core training pipeline."
_EXPORT_DEFAULT_HELP = "Destination JSONL file (default: <atlas-core-path>/exports/<timestamp>.jsonl)."
_DATABASE_DEFAULT_HELP = (
    "PostgreSQL connection URL. Defaults to STORAGE__DATABASE_URL or DATABASE_URL when exporting."
)
_SAMPLE_DATASET_PACKAGE = "atlas.data"
_SAMPLE_DATASET_NAME = "sample_traces.jsonl"
_SENSITIVE_MARKERS = ("key", "token", "secret", "password")


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register the `atlas train` subcommand."""

    parser = subparsers.add_parser("train", help=_TRAIN_HELP)
    add_export_arguments(
        parser,
        require_database_url=False,
        database_url_default=None,
        database_help=_DATABASE_DEFAULT_HELP,
        require_output=False,
        output_default=None,
        output_help=_EXPORT_DEFAULT_HELP,
    )
    parser.add_argument(
        "--atlas-core-path",
        default=None,
        help="Path to the Atlas Core repository (defaults to ATLAS_CORE_PATH).",
    )
    parser.add_argument("--config-name", default=None, help="Override Hydra config name passed to Atlas Core.")
    parser.add_argument("--data-config", default=None, help="Override data config for Hydra.")
    parser.add_argument("--trainer-config", default=None, help="Override trainer config for Hydra.")
    parser.add_argument("--model-config", default=None, help="Override model config for Hydra.")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="Fraction of samples reserved for evaluation (forwarded to Atlas Core).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of samples passed to Atlas Core.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory override for Atlas Core checkpoints and logs.",
    )
    parser.add_argument("--wandb-project", default=None, help="Weights & Biases project name override.")
    parser.add_argument("--wandb-run-name", default=None, help="Weights & Biases run name.")
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=None,
        help="Hydra override string. Repeat to supply multiple overrides.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip launching Atlas Core and only print the training command.",
    )
    parser.add_argument(
        "--use-sample-dataset",
        action="store_true",
        help="Copy the bundled sample dataset instead of exporting from Postgres.",
    )
    parser.set_defaults(handler=_cmd_train)
    return parser


def _resolve_atlas_core_path(raw: str | None) -> Path | None:
    """Resolve the Atlas Core path from CLI arguments or environment."""

    candidate = raw or os.environ.get("ATLAS_CORE_PATH")
    if not candidate:
        return None
    return Path(candidate).expanduser().resolve()


def _ensure_atlas_core_layout(path: Path) -> bool:
    """Validate that the Atlas Core repository includes the required scripts."""

    if not path.exists() or not path.is_dir():
        return False
    train_py = path / "train.py"
    pipeline_script = path / "scripts" / "run_offline_pipeline.py"
    return train_py.is_file() and pipeline_script.is_file()


def _default_export_path(atlas_core_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return atlas_core_path / "exports" / f"{timestamp}.jsonl"


def _resolve_database_url(cli_value: str | None) -> str | None:
    return cli_value or os.environ.get("STORAGE__DATABASE_URL") or os.environ.get("DATABASE_URL")


def _build_training_command(args: argparse.Namespace, export_path: Path) -> list[str]:
    command: list[str] = [sys.executable, "scripts/run_offline_pipeline.py", "--export-path", str(export_path)]
    option_map: Sequence[tuple[str, str]] = (
        ("config_name", "--config-name"),
        ("data_config", "--data-config"),
        ("trainer_config", "--trainer-config"),
        ("model_config", "--model-config"),
        ("eval_ratio", "--eval-ratio"),
        ("max_samples", "--max-samples"),
        ("output_dir", "--output-dir"),
        ("wandb_project", "--wandb-project"),
        ("wandb_run_name", "--wandb-run-name"),
    )
    for attr, flag in option_map:
        value = getattr(args, attr, None)
        if value is not None:
            command.extend([flag, str(value)])
    overrides: Iterable[str] | None = getattr(args, "overrides", None)
    if overrides:
        for override in overrides:
            command.extend(["--override", override])
    return command


def _redact_token(token: str) -> str:
    lowered = token.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "<redacted>"
    return token


def _redact_command(tokens: Sequence[str]) -> list[str]:
    return [_redact_token(token) for token in tokens]


def _copy_sample_dataset(destination: Path) -> None:
    """Copy the bundled sample dataset to the requested location."""

    try:
        dataset = resources.files(_SAMPLE_DATASET_PACKAGE).joinpath(_SAMPLE_DATASET_NAME)
    except (ModuleNotFoundError, FileNotFoundError):
        raise FileNotFoundError(
            "Sample dataset is unavailable. Ensure the Atlas SDK package includes bundled data resources."
        ) from None

    if not dataset.is_file():
        raise FileNotFoundError(
            f"Sample dataset {_SAMPLE_DATASET_NAME} is missing from package {_SAMPLE_DATASET_PACKAGE}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with dataset.open("rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def _run_training(atlas_core_path: Path, command: Sequence[str]) -> int:
    try:
        result = subprocess.run(command, cwd=str(atlas_core_path), check=False)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard for missing python executable/scripts
        print(f"Failed to launch Atlas Core training: {exc}", file=sys.stderr)
        return 1
    return result.returncode


def _cmd_train(args: argparse.Namespace) -> int:
    atlas_core_path = _resolve_atlas_core_path(args.atlas_core_path)
    if atlas_core_path is None or not _ensure_atlas_core_layout(atlas_core_path):
        print(
            "Atlas Core repository not found or incomplete. Clone Arc-Computer/ATLAS "
            "and set ATLAS_CORE_PATH or pass --atlas-core-path.",
            file=sys.stderr,
        )
        return 2

    export_path = Path(args.output).expanduser().resolve() if args.output else _default_export_path(atlas_core_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_sample_dataset:
        try:
            _copy_sample_dataset(export_path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        summary = None
    else:
        database_url = _resolve_database_url(args.database_url)
        if not database_url:
            print(
                "A database URL is required to export sessions. Provide --database-url or set "
                "STORAGE__DATABASE_URL / DATABASE_URL.",
                file=sys.stderr,
            )
            return 2
        configure_logging(args.quiet)
        review_status_filters, include_all = resolve_review_filters(
            args.config,
            args.include_review_statuses,
            args.include_all_statuses,
        )

        request = ExportRequest(
            database_url=database_url,
            output_path=export_path,
            session_ids=args.session_ids,
            limit=args.limit,
            offset=args.offset,
            status_filters=args.statuses,
            review_status_filters=review_status_filters,
            include_all_review_statuses=include_all,
            trajectory_event_limit=args.trajectory_event_limit,
            batch_size=args.batch_size,
        )
        summary = export_sessions_sync(request)
        if summary.sessions == 0:
            print(
                "Export completed but no sessions were written. Check your filters before launching training.",
                file=sys.stderr,
            )

    command = _build_training_command(args, export_path)
    if args.dry_run:
        command_str = shlex.join(command)
        print(f"Atlas Core command (dry-run): {command_str}")
        print("Dry run enabled; skipping Atlas Core execution.")
        print(f"Dataset available at: {export_path}")
        return 0
    else:
        redacted_command = shlex.join(_redact_command(command))
        print(f"Atlas Core command: {redacted_command}")

    exit_code = _run_training(atlas_core_path, command)
    if exit_code != 0:
        return exit_code

    print(f"Export written to: {export_path}")
    checkpoints_hint = args.output_dir or (atlas_core_path / "outputs")
    print(f"Atlas Core checkpoints will appear under: {checkpoints_hint}")
    return 0


__all__ = ["register_parser"]
