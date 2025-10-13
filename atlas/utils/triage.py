"""Utilities for building structured triage dossiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


Severity = Literal["low", "moderate", "high", "critical"]


class TriageSignal(BaseModel):
    """Lightweight diagnostic signal surfaced during triage."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: Any
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    annotation: str | None = None


class TriageRisk(BaseModel):
    """Risk factor captured during triage."""

    model_config = ConfigDict(extra="forbid")

    category: str = "general"
    description: str
    severity: Severity = "moderate"


class PersonaReference(BaseModel):
    """Link to an existing persona memory entry that may be relevant."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    rationale: str | None = None
    weight: float = Field(default=1.0, ge=0.0, le=5.0)
    tags: List[str] = Field(default_factory=list)


class EmbeddingVector(BaseModel):
    """Embedding payload attached to the triage dossier."""

    model_config = ConfigDict(extra="forbid")

    vector: List[float]
    model: str | None = None
    note: str | None = None


class TriageDossier(BaseModel):
    """Structured dossier describing the incident / task context."""

    model_config = ConfigDict(extra="forbid")

    task: str
    summary: str
    risks: List[TriageRisk] = Field(default_factory=list)
    signals: List[TriageSignal] = Field(default_factory=list)
    persona_references: List[PersonaReference] = Field(default_factory=list)
    embeddings: Dict[str, EmbeddingVector] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    fingerprint_hint: str | None = None


@dataclass
class TriageDossierBuilder:
    """Helper for incrementally constructing a TriageDossier."""

    task: str
    summary: str | None = None
    fingerprint_hint: str | None = None
    tags: List[str] = None  # type: ignore[assignment]
    _risks: List[TriageRisk] = None  # type: ignore[assignment]
    _signals: List[TriageSignal] = None  # type: ignore[assignment]
    _personas: List[PersonaReference] = None  # type: ignore[assignment]
    _embeddings: Dict[str, EmbeddingVector] = None  # type: ignore[assignment]
    _metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.tags = list(self.tags or [])
        self._risks = list(self._risks or [])
        self._signals = list(self._signals or [])
        self._personas = list(self._personas or [])
        self._embeddings = dict(self._embeddings or {})
        self._metadata = dict(self._metadata or {})

    def set_summary(self, summary: str) -> "TriageDossierBuilder":
        self.summary = summary.strip()
        return self

    def add_risk(
        self,
        description: str,
        *,
        category: str = "general",
        severity: Severity = "moderate",
    ) -> "TriageDossierBuilder":
        self._risks.append(TriageRisk(category=category, description=description, severity=severity))
        return self

    def add_signal(
        self,
        name: str,
        value: Any,
        *,
        confidence: float | None = None,
        annotation: str | None = None,
    ) -> "TriageDossierBuilder":
        self._signals.append(
            TriageSignal(
                name=name,
                value=value,
                confidence=confidence,
                annotation=annotation,
            )
        )
        return self

    def add_persona_reference(
        self,
        persona_id: str,
        *,
        rationale: str | None = None,
        weight: float = 1.0,
        tags: Sequence[str] | None = None,
    ) -> "TriageDossierBuilder":
        self._personas.append(
            PersonaReference(
                persona_id=persona_id,
                rationale=rationale,
                weight=weight,
                tags=list(tags or []),
            )
        )
        return self

    def add_embedding(
        self,
        key: str,
        vector: Iterable[float],
        *,
        model: str | None = None,
        note: str | None = None,
    ) -> "TriageDossierBuilder":
        self._embeddings[key] = EmbeddingVector(vector=[float(v) for v in vector], model=model, note=note)
        return self

    def add_tags(self, *tags: str) -> "TriageDossierBuilder":
        for tag in tags:
            normalised = tag.strip()
            if normalised and normalised not in self.tags:
                self.tags.append(normalised)
        return self

    def update_metadata(self, **values: Any) -> "TriageDossierBuilder":
        self._metadata.update(values)
        return self

    def build(self) -> TriageDossier:
        summary = (self.summary or self.task).strip()
        return TriageDossier(
            task=self.task,
            summary=summary,
            risks=list(self._risks),
            signals=list(self._signals),
            persona_references=list(self._personas),
            embeddings=dict(self._embeddings),
            tags=list(self.tags),
            metadata=dict(self._metadata),
            fingerprint_hint=self.fingerprint_hint,
        )


def attach_triage_to_context(context: "ExecutionContext", dossier: TriageDossier) -> None:
    """Attach the triage dossier to the execution context metadata."""

    context.metadata["triage"] = {
        "dossier": dossier.model_dump(),
    }


__all__ = [
    "attach_triage_to_context",
    "EmbeddingVector",
    "PersonaReference",
    "Severity",
    "TriageDossier",
    "TriageDossierBuilder",
    "TriageRisk",
    "TriageSignal",
]
