"""Prompt rewrite engine that derives planner/teacher personas via LLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from atlas.config.models import AdapterConfig, PromptRewriteConfig, StudentConfig, TeacherConfig
from atlas.config.models import LLMParameters
from atlas.utils.llm_client import LLMClient


@dataclass(frozen=True)
class RewrittenStudentPrompts:
    planner: str
    executor: str
    synthesizer: str


@dataclass(frozen=True)
class RewrittenTeacherPrompts:
    plan_review: str
    validation: str
    guidance: str


@dataclass(frozen=True)
class PersonaSpec:
    family: str
    output_key: str
    title: str
    focus: str
    deliverables: List[str]
    constraints: List[str]


class PromptRewriteEngine:
    """Generates specialised prompts for student/teacher personas using an LLM."""

    def __init__(
        self,
        config: PromptRewriteConfig | None,
        fallback_llm: LLMParameters | None,
    ) -> None:
        self._max_tokens = (config.max_tokens if config else 1024)
        self._temperature = (config.temperature if config else 0.1)
        self._llm_params = (config.llm if config and config.llm is not None else fallback_llm)
        if self._llm_params is None:
            raise ValueError(
                "Prompt rewrite requires an LLM configuration; provide prompt_rewrite.llm in the config"
            )
        self._client = LLMClient(self._llm_params)
        self._cache: dict[str, tuple[RewrittenStudentPrompts, RewrittenTeacherPrompts]] = {}

    async def generate(
        self,
        base_prompt: str,
        adapter_config: AdapterConfig,
        student_config: StudentConfig,
        teacher_config: TeacherConfig,
    ) -> tuple[RewrittenStudentPrompts, RewrittenTeacherPrompts]:
        context = self._build_context(base_prompt, adapter_config, student_config, teacher_config)
        cache_key = self._cache_key(context)
        if cache_key in self._cache:
            return self._cache[cache_key]
        specs = self._persona_specs()
        persona_prompts = await asyncio.gather(
            *(self._rewrite_persona(context, spec) for spec in specs)
        )
        student_mapping: Dict[str, str] = {}
        teacher_mapping: Dict[str, str] = {}
        for spec, prompt_text in zip(specs, persona_prompts):
            if spec.family == "student":
                student_mapping[spec.output_key] = prompt_text
            else:
                teacher_mapping[spec.output_key] = prompt_text

        try:
            student_prompts = RewrittenStudentPrompts(
                planner=student_mapping["planner"],
                executor=student_mapping["executor"],
                synthesizer=student_mapping["synthesizer"],
            )
            teacher_prompts = RewrittenTeacherPrompts(
                plan_review=teacher_mapping["plan_review"],
                validation=teacher_mapping["validation"],
                guidance=teacher_mapping["guidance"],
            )
        except KeyError as exc:
            missing = str(exc)
            raise ValueError(f"Prompt rewrite missing persona output {missing}") from exc

        prompts = (student_prompts, teacher_prompts)
        self._cache[cache_key] = prompts
        return prompts

    def _build_context(
        self,
        base_prompt: str,
        adapter_config: AdapterConfig,
        student_config: StudentConfig,
        teacher_config: TeacherConfig,
    ) -> Dict[str, Any]:
        tool_catalog = []
        for tool in getattr(adapter_config, "tools", []) or []:
            tool_catalog.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_dump(by_alias=True),
                    "output_schema": tool.output_schema,
                }
            )

        adapter_snapshot = {
            "type": getattr(getattr(adapter_config, "type", None), "value", None),
            "name": getattr(adapter_config, "name", None),
        }

        student_settings = {
            "tool_choice": student_config.tool_choice,
            "guidance": dict(student_config.prompt_guidance or {}),
        }

        teacher_settings = {
            "plan_cache_seconds": teacher_config.plan_cache_seconds,
            "guidance": dict(teacher_config.prompt_guidance or {}),
        }

        return {
            "base_agent_prompt": base_prompt,
            "adapter": adapter_snapshot,
            "tool_catalog": tool_catalog,
            "student": student_settings,
            "teacher": teacher_settings,
        }

    def _persona_specs(self) -> List[PersonaSpec]:
        return [
            PersonaSpec(
                family="student",
                output_key="planner",
                title="Planner Student",
                focus=(
                    "Transform tasks into an ordered JSON plan with dependencies."
                    " Ensure independent steps are clearly tagged so the orchestrator"
                    " can execute them in parallel while respecting prerequisites."
                ),
                deliverables=[
                    "Structured plan enumerating steps with ids, descriptions, dependencies, and tool usage recommendations",
                    "Explicit parallelisation cues highlighting which steps may run concurrently",
                    "Pre-flight validations covering resource, compliance, and context requirements",
                ],
                constraints=[
                    "Must not drop user constraints or domain-specific guardrails from the base prompt",
                    "Identify prerequisites for every step and call out missing data or permissions",
                    "Avoid unapproved tools or actions that violate compliance or platform policies",
                ],
            ),
            PersonaSpec(
                family="student",
                output_key="executor",
                title="Executor Student",
                focus=(
                    "Execute a single plan step using available tools and prior context."
                    " Instructions must remain correct even when multiple independent"
                    " steps execute simultaneously in separate threads of work."
                ),
                deliverables=[
                    "Detailed execution methodology referencing approved tools and inputs",
                    "Concurrency-safe context handling and conflict resolution guidance",
                    "Structured logging expectations for telemetry and audits",
                ],
                constraints=[
                    "Never invoke disallowed operations or unreviewed integrations",
                    "Respect rate limits, data classification, and compliance policies",
                    "Escalate or pause when prerequisites from peer steps are missing",
                ],
            ),
            PersonaSpec(
                family="student",
                output_key="synthesizer",
                title="Synthesizer Student",
                focus=(
                    "Aggregate completed step outputs into a final response."
                    " Honour every constraint from the base agent prompt and cite"
                    " the contributing steps while reconciling concurrent results."
                ),
                deliverables=[
                    "Structured summary that references contributing step ids and evidence",
                    "Risk, uncertainty, and follow-up sections for unresolved items",
                    "Format guidance for citations, provenance, and final answer shape",
                ],
                constraints=[
                    "Final output must maintain tone and formatting requirements from the base prompt",
                    "Do not manufacture facts; rely only on validated step outputs or flag gaps",
                    "Include remediation or escalation instructions when success criteria are unmet",
                ],
            ),
            PersonaSpec(
                family="teacher",
                output_key="plan_review",
                title="Plan Review Teacher",
                focus=(
                    "Audit the student plan for completeness, correctness, and"
                    " concurrency readiness. Enforce valid dependencies so only"
                    " independent steps run in parallel."
                ),
                deliverables=[
                    "Assessment rubric covering completeness, dependency integrity, risk, and compliance",
                    "Rewritten plan when deficiencies are found, maintaining original context",
                    "Approval or rejection decision with justification",
                ],
                constraints=[
                    "Cache results according to plan_cache_seconds to avoid redundant approvals",
                    "Spot and forbid hidden circular dependencies or missing prerequisites",
                    "Flag requirements for human review when risks exceed thresholds",
                ],
            ),
            PersonaSpec(
                family="teacher",
                output_key="validation",
                title="Validation Teacher",
                focus=(
                    "Validate execution traces and outputs for each step,"
                    " considering that peer steps may complete in parallel."
                    " Return structured judgements aligned with the RIM judges."
                ),
                deliverables=[
                    "Deterministic {\"valid\": bool, \"rationale\": str} assessments aligned with orchestration schema",
                    "Callouts for concurrency conflicts, missing evidence, or safety breaches",
                    "Signal data for RIM judges including process and helpfulness cues",
                ],
                constraints=[
                    "Produce actionable rationales that downstream systems can parse",
                    "Reference telemetry traces, tool outputs, and guidance history",
                    "Escalate when evidence is insufficient or policies are violated",
                ],
            ),
            PersonaSpec(
                family="teacher",
                output_key="guidance",
                title="Guidance Teacher",
                focus=(
                    "Produce concise remediation guidance that helps the executor"
                    " correct mistakes on the next retry while remaining aware of"
                    " concurrent workstreams."
                ),
                deliverables=[
                    "Targeted remediation steps prioritised for the next attempt",
                    "Tool usage or research recommendations gated by policy",
                    "Signals for when to request human assistance or expand scope",
                ],
                constraints=[
                    "Avoid duplicating stale advice and reference the most recent teacher feedback",
                    "Highlight dependencies on parallel steps and suggest coordination tactics",
                    "Maintain tone consistent with the base prompt and organisational policies",
                ],
            ),
        ]

    async def _rewrite_persona(self, context: Dict[str, Any], spec: PersonaSpec) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You design production-grade system instructions for an Atlas persona."
                    " Always preserve the provided base_agent_prompt verbatim at the top of"
                    " the final prompt. Append persona-specific guidance that is exhaustive"
                    " enough for enterprise workflows."
                    "\n\n"
                    "The appended guidance must follow this format, using uppercase section"
                    " titles:"
                    "\n1. ROLE OVERVIEW"
                    "\n2. PRIMARY OBJECTIVES"
                    "\n3. EXECUTION DIRECTIVES"
                    "\n4. SAFETY AND COMPLIANCE"
                    "\n5. COLLABORATION AND HANDOFFS"
                    "\n6. QUALITY BAR"
                    "\n7. FAILURE AND ESCALATION"
                    "\nEach section should contain clear sentences or numbered directives."
                    " Incorporate every deliverable and constraint provided. Reflect the"
                    " persona focus, concurrency guarantees, tool catalog, and token limits."
                    " Do not emit markdown fences, JSON, or bullet characters other than"
                    " numbers or dashes where appropriate."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(self._persona_payload(context, spec), ensure_ascii=False, indent=2),
            },
        ]
        response = await self._client.acomplete(
            messages,
            overrides={"max_tokens": self._max_tokens, "temperature": self._temperature},
        )
        addition = response.content.strip()
        if not addition:
            raise ValueError(f"Prompt rewrite produced empty instructions for {spec.title}")
        base_prompt = context["base_agent_prompt"].strip()
        if addition.startswith(base_prompt):
            trimmed = addition[len(base_prompt):].lstrip()
        else:
            trimmed = addition
        if base_prompt and trimmed:
            combined = f"{base_prompt}\n\n{trimmed}"
        else:
            combined = base_prompt or trimmed
        return combined

    def _persona_payload(self, context: Dict[str, Any], spec: PersonaSpec) -> Dict[str, Any]:
        if spec.family == "student":
            guidance = context["student"]["guidance"].get(spec.output_key)
        else:
            guidance = context["teacher"]["guidance"].get(spec.output_key)

        return {
            "persona": {
                "family": spec.family,
                "output_key": spec.output_key,
                "title": spec.title,
                "focus": spec.focus,
                "additional_guidance": guidance,
                "deliverables": spec.deliverables,
                "constraints": spec.constraints,
            },
            "base_agent_prompt": context["base_agent_prompt"],
            "adapter": context["adapter"],
            "tool_catalog": context["tool_catalog"],
            "student_settings": context["student"],
            "teacher_settings": context["teacher"],
            "concurrency_requirements": {
                "independent_steps_may_execute_in_parallel": True,
                "expect_exact_dependency_tracking": True,
            },
            "output_expectation": {
                "format": "plain_text",
                "must_preserve_base_prompt": True,
                "forbidden_styles": ["json", "markdown_fence"],
            },
        }

    def _cache_key(self, context: Dict[str, Any]) -> str:
        digest = hashlib.sha256()
        serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
        digest.update(serialized.encode("utf-8"))
        return digest.hexdigest()


__all__ = [
    "PromptRewriteEngine",
    "RewrittenStudentPrompts",
    "RewrittenTeacherPrompts",
]
