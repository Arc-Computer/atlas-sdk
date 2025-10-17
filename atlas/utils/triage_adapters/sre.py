"""Reference triage adapter for SRE / incident response workflows."""

from __future__ import annotations

from typing import Any, Dict

from atlas.utils.triage import TriageDossier, TriageDossierBuilder


def build_dossier(task: str, metadata: Dict[str, Any] | None = None) -> TriageDossier:
    """Construct a triage dossier for SRE incidents."""

    metadata = metadata or {}
    incident = metadata.get("incident") or {}
    service = incident.get("service") or metadata.get("service") or "unknown-service"
    impact = incident.get("impact") or metadata.get("impact") or "unknown"
    recent_changes = metadata.get("recent_changes", [])
    alerts = metadata.get("alerts", [])

    builder = TriageDossierBuilder(task=task, fingerprint_hint=f"sre::{service}")
    builder.set_summary(f"SRE investigation for {service}: {impact} impact.")
    builder.add_tags("domain:sre", f"service:{service}")
    builder.update_metadata(service=service, impact=impact, alerts=alerts)

    if alerts:
        builder.add_signal(
            "alerts.active_count",
            len(alerts),
            annotation="Number of active alerts referencing this service.",
        )
    for change in recent_changes[:3]:
        builder.add_signal("recent_change", change, annotation="Recent deploy or configuration change.")

    builder.add_risk(
        description="Customer experience degradation if MTTR exceeds SLA.",
        category="customer",
        severity="high",
    )
    builder.add_risk(
        description="Potential cascading failure across dependent services.",
        category="infrastructure",
        severity="moderate",
    )

    embeddings = metadata.get("embeddings", {})
    for key, value in embeddings.items():
        vector = value.get("vector") if isinstance(value, dict) else value
        if isinstance(vector, (list, tuple)):
            builder.add_embedding(key, vector, model=value.get("model") if isinstance(value, dict) else None)

    return builder.build()


__all__ = ["build_dossier"]
