from __future__ import annotations

import os
from pathlib import Path

from atlas.cli import main as atlas_cli_main  # type: ignore[attr-defined]
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.triage import (
    TriageDossierBuilder,
    attach_triage_to_context,
)
from atlas.utils.triage_adapters import sre, support, code


def test_builder_constructs_minimal_dossier():
    builder = (
        TriageDossierBuilder(task="Investigate payment outage")
        .set_summary("Payment API returning 500s")
        .add_risk("MTTR breach risk", severity="high")
        .add_signal("alert.count", 3, confidence=0.8)
        .add_persona_reference("persona-123", rationale="Previous fix for similar issue", weight=1.1)
    )
    dossier = builder.build()
    assert dossier.summary == "Payment API returning 500s"
    assert dossier.risks[0].severity == "high"
    assert dossier.signals[0].name == "alert.count"
    assert dossier.persona_references[0].persona_id == "persona-123"


def test_reference_adapters_attach_domain_tags():
    metadata = {
        "incident": {"service": "payments", "impact": "high"},
        "alerts": [{"name": "critical_cpu"}],
        "recent_changes": ["deploy abc123"],
    }
    dossier = sre.build_dossier("Investigate payment outage", metadata=metadata)
    assert "domain:sre" in dossier.tags
    assert dossier.metadata["service"] == "payments"

    support_dossier = support.build_dossier("Assist customer", metadata={"customer": {"name": "Acme", "tier": "enterprise"}})
    assert "domain:support" in support_dossier.tags

    code_dossier = code.build_dossier("Fix failing tests", metadata={"repo": "atlas", "failing_tests": ["test_core"]})
    assert "domain:code" in code_dossier.tags
    assert code_dossier.signals[0].name == "failing_tests"


def test_attach_triage_to_context_sets_metadata():
    context = ExecutionContext.get()
    context.reset()
    dossier = TriageDossierBuilder(task="Test task").set_summary("summary").build()
    attach_triage_to_context(context, dossier)
    assert context.metadata["triage"]["dossier"]["summary"] == "summary"


def test_cli_triage_init_creates_file(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "adapter.py"
    exit_code = atlas_cli_main.main(["triage", "init", "--domain", "sre", "--output", str(output_path)])
    assert exit_code == 0
    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "TriageDossierBuilder" in contents
