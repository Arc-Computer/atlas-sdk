"""Reference triage adapter for customer support investigations."""

from __future__ import annotations

from typing import Any, Dict

from atlas.utils.triage import TriageDossier, TriageDossierBuilder


def build_dossier(task: str, metadata: Dict[str, Any] | None = None) -> TriageDossier:
    metadata = metadata or {}
    customer = metadata.get("customer") or {}
    account_tier = customer.get("tier") or "standard"
    sentiment = metadata.get("sentiment", "neutral")
    product_area = metadata.get("product_area") or "general"

    builder = TriageDossierBuilder(task=task, fingerprint_hint=f"support::{product_area}")
    builder.set_summary(f"Support request in {product_area} from {customer.get('name', 'unknown customer')}.")
    builder.add_tags("domain:support", f"tier:{account_tier}", f"product:{product_area}")

    builder.add_signal("customer.tier", account_tier)
    builder.add_signal("customer.sentiment", sentiment, annotation="Latest NLU sentiment score for the conversation.")

    if metadata.get("csat_history"):
        builder.add_signal(
            "customer.csat_history",
            metadata["csat_history"],
            annotation="Rolling CSAT trend for the account.",
        )

    builder.add_risk(
        description="Customer churn risk if issue is not resolved within SLA.",
        category="customer",
        severity="high" if account_tier in {"enterprise", "premium"} else "moderate",
    )
    builder.add_risk(
        description="Potential reputational harm due to negative sentiment on public channels.",
        category="reputation",
        severity="moderate",
    )

    for persona in metadata.get("knowledge_base_hits", []):
        if isinstance(persona, dict):
            builder.add_persona_reference(
                persona_id=persona.get("id", ""),
                rationale=persona.get("excerpt"),
                weight=float(persona.get("score", 1.0)),
                tags=persona.get("tags"),
            )
        elif isinstance(persona, str):
            builder.add_persona_reference(persona, rationale="Relevant knowledge base entry.", weight=0.9)

    return builder.build()


__all__ = ["build_dossier"]
