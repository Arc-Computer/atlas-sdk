"""Reference triage adapter for code-review / debugging workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from atlas.utils.triage import TriageDossier, TriageDossierBuilder


def _format_modules(modules: Iterable[str]) -> str:
    normalized = [module.strip() for module in modules if module]
    return ", ".join(sorted(set(normalized))) or "unknown"


def build_dossier(task: str, metadata: Dict[str, Any] | None = None) -> TriageDossier:
    metadata = metadata or {}
    repo = metadata.get("repo") or "unknown-repo"
    modules = metadata.get("modules", [])
    failing_tests = metadata.get("failing_tests", [])
    owner = metadata.get("owner") or "unassigned"

    builder = TriageDossierBuilder(task=task, fingerprint_hint=f"code::{repo}")
    builder.set_summary(f"Code triage for {repo}, owner {owner}.")
    builder.add_tags("domain:code", f"repo:{repo}", f"owner:{owner}")

    if modules:
        builder.add_signal("modules", modules, annotation="Modules touched by recent commits.")

    if failing_tests:
        builder.add_signal(
            "failing_tests",
            failing_tests,
            annotation="Tests failing in the latest CI run.",
        )
        builder.add_risk(
            description="Persistent CI failures block deployments.",
            category="quality",
            severity="high",
        )

    builder.add_risk(
        description="Regression risk if coverage gaps remain.",
        category="quality",
        severity="moderate",
    )

    for persona in metadata.get("persona_recommendations", []):
        if isinstance(persona, dict):
            builder.add_persona_reference(
                persona_id=persona.get("id", ""),
                rationale=persona.get("reason"),
                weight=float(persona.get("weight", 1.0)),
                tags=persona.get("tags"),
            )
        elif isinstance(persona, str):
            builder.add_persona_reference(persona, rationale="Historical fix for similar failure.", weight=1.1)

    if metadata.get("embedding"):
        embedding = metadata["embedding"]
        if isinstance(embedding, dict) and isinstance(embedding.get("vector"), (list, tuple)):
            builder.add_embedding(
                "code_context",
                embedding["vector"],
                model=embedding.get("model"),
                note="Code context embedding for retrieval.",
            )

    return builder.build()


__all__ = ["build_dossier"]
