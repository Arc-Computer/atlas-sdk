"""Shared dataclasses representing runtime reward and trace payloads."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AtlasJudgeSample:
    """Fine-grained sample emitted by a reward judge."""

    score: float
    rationale: str
    principles: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Optional[float] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AtlasJudgeSample":
        return cls(
            score=float(payload.get("score", 0.0)),
            rationale=payload.get("rationale", ""),
            principles=list(payload.get("principles", []) or []),
            uncertainty=payload.get("uncertainty"),
            temperature=payload.get("temperature"),
        )


@dataclass
class AtlasJudgeBreakdown:
    """Structured result from a single reward judge."""

    identifier: str
    score: float
    rationale: str
    principles: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[AtlasJudgeSample] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "score": self.score,
            "rationale": self.rationale,
            "principles": self.principles,
            "samples": [sample.to_dict() for sample in self.samples],
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AtlasJudgeBreakdown":
        samples_payload = payload.get("samples") or []
        samples = [
            AtlasJudgeSample.from_dict(sample) if isinstance(sample, dict) else sample
            for sample in samples_payload
        ]
        return cls(
            identifier=payload.get("identifier", ""),
            score=float(payload.get("score", 0.0)),
            rationale=payload.get("rationale", ""),
            principles=list(payload.get("principles", []) or []),
            samples=samples,
            escalated=bool(payload.get("escalated", False)),
            escalation_reason=payload.get("escalation_reason"),
        )


@dataclass
class AtlasRewardBreakdown:
    """Aggregated reward summary for a step or episode."""

    score: float
    judges: List[AtlasJudgeBreakdown] = field(default_factory=list)
    rationale: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "judges": [judge.to_dict() for judge in self.judges],
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AtlasRewardBreakdown":
        judges_payload = payload.get("judges") or []
        judges = [
            AtlasJudgeBreakdown.from_dict(judge) if isinstance(judge, dict) else judge
            for judge in judges_payload
        ]
        return cls(
            score=float(payload.get("score", 0.0)),
            judges=judges,
            rationale=payload.get("rationale"),
            raw=payload,
        )


@dataclass
class AtlasStepTrace:
    """Single plan step with execution, validation, and reward context."""

    step_id: int
    description: str
    trace: str
    output: str
    reward: AtlasRewardBreakdown
    tool: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    guidance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    deliverable: Any | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "trace": self.trace,
            "output": self.output,
            "reward": self.reward.to_dict(),
            "tool": self.tool,
            "tool_params": self.tool_params,
            "context": self.context,
            "validation": self.validation,
            "attempts": self.attempts,
            "guidance": self.guidance,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "deliverable": self.deliverable,
        }


@dataclass
class AtlasSessionTrace:
    """Complete session exported from the runtime."""

    task: str
    final_answer: str
    plan: Dict[str, Any]
    steps: List[AtlasStepTrace]
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "final_answer": self.final_answer,
            "plan": self.plan,
            "steps": [step.to_dict() for step in self.steps],
            "session_metadata": self.session_metadata,
        }
