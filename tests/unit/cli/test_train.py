from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import pytest

from atlas.cli import main as cli_main
from atlas.cli import train as train_cli
from atlas.cli.jsonl_writer import ExportSummary


def _make_atlas_core_tree(base: Path) -> Path:
    atlas_core = base / "atlas-core"
    scripts_dir = atlas_core / "scripts"
    scripts_dir.mkdir(parents=True)
    (atlas_core / "train.py").write_text("# stub\n", encoding="utf-8")
    (scripts_dir / "run_offline_pipeline.py").write_text("# stub\n", encoding="utf-8")
    return atlas_core


def test_train_missing_atlas_core_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli_main.build_parser()
    monkeypatch.delenv("ATLAS_CORE_PATH", raising=False)
    args = parser.parse_args(["train"])

    exit_code = args.handler(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Atlas Core repository not found" in captured.err


def test_train_dry_run_invokes_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    atlas_core = _make_atlas_core_tree(tmp_path)
    dataset_path = tmp_path / "export" / "traces.jsonl"
    parser = cli_main.build_parser()

    export_calls: list = []

    def fake_export(request):
        export_calls.append(request)
        return ExportSummary(sessions=3, steps=7)

    monkeypatch.setenv("STORAGE__DATABASE_URL", "postgresql://atlas:atlas@localhost:5433/atlas")
    monkeypatch.setattr(train_cli, "export_sessions_sync", fake_export)
    def dry_run_guard(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called during dry-run")

    monkeypatch.setattr(train_cli.subprocess, "run", dry_run_guard)
    monkeypatch.setattr(train_cli.sys, "executable", "python")

    args = parser.parse_args(
        [
            "train",
            "--atlas-core-path",
            str(atlas_core),
            "--output",
            str(dataset_path),
            "--dry-run",
            "--config-name",
            "baseline",
            "--override",
            "trainer.lr=1e-4",
        ]
    )

    exit_code = args.handler(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert export_calls, "export_sessions_sync should be invoked during dry-run"
    assert "Atlas Core command: python scripts/run_offline_pipeline.py" in captured.out
    assert "--config-name baseline" in captured.out
    assert "--override trainer.lr=1e-4" in captured.out
    assert "Dry run enabled; skipping Atlas Core execution." in captured.out


def test_train_happy_path_default_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    atlas_core = _make_atlas_core_tree(tmp_path)
    parser = cli_main.build_parser()

    fixed_timestamp = datetime(2024, 1, 2, 3, 4, 5)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return fixed_timestamp

    export_requests: list = []

    def fake_export(request):
        export_requests.append(request)
        request.output_path.write_text("{}", encoding="utf-8")
        return ExportSummary(sessions=2, steps=5)

    run_calls: list[types.SimpleNamespace] = []

    def fake_run(cmd, cwd=None, check=False):
        run_calls.append(types.SimpleNamespace(cmd=cmd, cwd=cwd, check=check))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setenv("ATLAS_CORE_PATH", str(atlas_core))
    monkeypatch.setenv("STORAGE__DATABASE_URL", "postgresql://atlas:atlas@localhost:5433/atlas")
    monkeypatch.setattr(train_cli, "datetime", FixedDateTime)
    monkeypatch.setattr(train_cli, "export_sessions_sync", fake_export)
    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)  # type: ignore[assignment]
    monkeypatch.setattr(train_cli.sys, "executable", "python")

    args = parser.parse_args(
        [
            "train",
            "--config-name",
            "baseline",
        ]
    )

    exit_code = args.handler(args)
    captured = capsys.readouterr()

    expected_output = atlas_core / "exports" / "20240102-030405.jsonl"

    assert exit_code == 0
    assert export_requests, "export_sessions_sync should be called"
    request = export_requests[0]
    assert request.output_path == expected_output
    assert expected_output.exists()

    assert run_calls, "subprocess.run should be invoked"
    run_call = run_calls[0]
    assert run_call.cmd[0] == "python"
    assert run_call.cmd[1:3] == ["scripts/run_offline_pipeline.py", "--export-path"]
    assert run_call.cmd[3] == str(expected_output)
    assert run_call.cwd == str(atlas_core)

    assert str(atlas_core / "outputs") in captured.out
