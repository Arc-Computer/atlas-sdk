from __future__ import annotations

from pathlib import Path

from atlas.cli import main as atlas_cli_main  # type: ignore[attr-defined]
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.triage import (
    TriageDossierBuilder,
    attach_triage_to_context,
    default_build_dossier,
)
from atlas.utils.triage_adapters import code, sre, support


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


def test_default_builder_heuristics():
    metadata = {
        "summary": "Investigate failing pipeline",
        "tags": ["domain:ml", "tenant:acme"],
        "risks": ["SLA breach", {"description": "Regression risk", "severity": "moderate"}],
        "signals": [{"name": "pipeline.status", "value": "failed"}],
        "persona_references": [{"persona_id": "persona-321", "rationale": "Last similar fix"}],
        "embeddings": {"ctx": {"vector": [0.1, 0.2, 0.3]}},
        "metadata": {"source": "alerting"},
        "custom_field": "value",
    }
    dossier = default_build_dossier("Investigate", metadata)
    assert dossier.summary == "Investigate failing pipeline"
    assert "domain:ml" in dossier.tags
    assert dossier.risks[0].description == "SLA breach"
    assert dossier.signals[0].name == "pipeline.status"
    assert dossier.metadata["custom_field"] == "value"
    assert dossier.persona_references[0].persona_id == "persona-321"


def test_attach_triage_to_context_sets_metadata():
    context = ExecutionContext.get()
    context.reset()
    dossier = TriageDossierBuilder(task="Test task").set_summary("summary").build()
    attach_triage_to_context(context, dossier)
    assert context.metadata["triage"]["dossier"]["summary"] == "summary"


def test_cli_triage_init_creates_file(tmp_path: Path):
    output_path = tmp_path / "adapter.py"
    exit_code = atlas_cli_main.main(["triage", "init", "--domain", "sre", "--output", str(output_path)])
    assert exit_code == 0
    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "TriageDossierBuilder" in contents
    # Ensure the generated adapter is syntactically valid.
    compile(contents, str(output_path), "exec")


def test_default_builder_normalises_unknown_severity():
    metadata = {
        "risks": [
            {"description": "Partner reported medium severity", "severity": "medium"},
            {"description": "Explicit sev2 mapping", "severity": "sev2"},
        ]
    }
    dossier = default_build_dossier("Investigate medium issue", metadata)
    severities = [risk.severity for risk in dossier.risks]
    assert severities == ["moderate", "moderate"]
