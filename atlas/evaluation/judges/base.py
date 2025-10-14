"""Compatibility shims for legacy per-step judge interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.types import Step


@dataclass
class JudgeSample:
    score: float
    rationale: str
    principles: List[Dict[str, Any]]
    uncertainty: float
    temperature: float
    reasoning: Dict[str, Any] | None = None


@dataclass
class JudgeOutcome:
    identifier: str
    score: float
    rationale: str
    principles: List[Dict[str, Any]]
    samples: List[JudgeSample]
    escalated: bool
    escalation_reason: str | None
    reasoning: Dict[str, Any] | None = None


class Judge:
    """Legacy judge base class retained for compatibility in tests."""

    def __init__(self, identifier: str, client: Any | None = None) -> None:
        self.identifier = identifier
        self._client = client

    async def asample(self, context: "JudgeContext", temperature: float) -> JudgeSample | None:
        raise NotImplementedError

    async def ajudge(self, context: "JudgeContext") -> JudgeOutcome:
        raise NotImplementedError

    def build_meta_prompt(self, context: "JudgeContext", samples: List[JudgeSample], reason: str | None) -> str:
        raise NotImplementedError


@dataclass
class JudgeContext:
    """Inputs describing a single step evaluation request."""

    task: str
    step: Step
    trace: str
    output: str
    attempt: int = 1
    prior_results: Dict[int, Any] | None = None
    guidance: Sequence[str] | None = None
    reward_override: AtlasRewardBreakdown | Dict[str, Any] | None = None
    execution_mode: str | None = None
    focus_prompt: str | None = None
    teacher_intervened: bool = False
    status: str | None = None
    validation: Dict[str, Any] | None = None
