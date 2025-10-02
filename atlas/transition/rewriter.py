"""Prompt rewrite engine that derives planner/teacher personas via LLM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from atlas.config.models import (
    AdapterConfig,
    PromptRewriteConfig,
    StudentConfig,
    TeacherConfig,
)
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
        cache_key = self._cache_key(base_prompt, adapter_config, student_config, teacher_config)
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = self._build_payload(base_prompt, adapter_config, student_config, teacher_config)
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite system prompts for a multi-agent architecture. Preserve the user's intent and domain"
                    " instructions exactly, while producing specialised prompts for a planner Student, an executor"
                    " Student, and a Teacher reviewer. Always respond with JSON matching the requested schema."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        response = await self._client.acomplete(
            messages,
            response_format={"type": "json_object"},
            overrides={"max_tokens": self._max_tokens, "temperature": self._temperature},
        )

        prompts = self._parse_response(response.content)
        self._cache[cache_key] = prompts
        return prompts

    def _build_payload(
        self,
        base_prompt: str,
        adapter_config: AdapterConfig,
        student_config: StudentConfig,
        teacher_config: TeacherConfig,
    ) -> Dict[str, Any]:
        tools = []
        for tool in getattr(adapter_config, "tools", []) or []:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_dump(by_alias=True),
                }
            )

        student_instructions = student_config.prompts.model_dump() if getattr(student_config, "prompts", None) else {}
        teacher_instructions = teacher_config.prompts.model_dump() if getattr(teacher_config, "prompts", None) else {}

        return {
            "base_agent_prompt": base_prompt,
            "available_tools": tools,
            "student_prompt_guidance": student_instructions,
            "teacher_prompt_guidance": teacher_instructions,
            "expect_output_schema": {
                "student": ["planner", "executor", "synthesizer"],
                "teacher": ["plan_review", "validation", "guidance"],
            },
            "notes": (
                "Generate concise, fully-specified system prompts. Planner should focus on drafting dependency-aware"
                " plans; executor should follow plans with precise instructions for tool usage; teacher should review,"
                " validate, and provide remediation guidance. Maintain the tone and constraints of the base agent"
                " prompt."
            ),
        }

    def _parse_response(
        self,
        content: str,
    ) -> tuple[RewrittenStudentPrompts, RewrittenTeacherPrompts]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Prompt rewrite LLM response was not valid JSON") from exc

        try:
            student = data["student"]
            teacher = data["teacher"]
        except KeyError as exc:
            raise ValueError("Prompt rewrite JSON missing required sections 'student' or 'teacher'") from exc

        try:
            student_prompts = RewrittenStudentPrompts(
                planner=str(student["planner"]),
                executor=str(student["executor"]),
                synthesizer=str(student["synthesizer"]),
            )
            teacher_prompts = RewrittenTeacherPrompts(
                plan_review=str(teacher["plan_review"]),
                validation=str(teacher["validation"]),
                guidance=str(teacher["guidance"]),
            )
        except KeyError as exc:
            raise ValueError("Prompt rewrite JSON missing expected keys") from exc

        return student_prompts, teacher_prompts

    def _cache_key(
        self,
        base_prompt: str,
        adapter_config: AdapterConfig,
        student_config: StudentConfig,
        teacher_config: TeacherConfig,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(base_prompt.strip().encode("utf-8"))
        digest.update(json.dumps([tool.name for tool in getattr(adapter_config, "tools", []) or []]).encode("utf-8"))
        if getattr(student_config, "prompts", None):
            digest.update(json.dumps(student_config.prompts.model_dump(), sort_keys=True).encode("utf-8"))
        if getattr(teacher_config, "prompts", None):
            digest.update(json.dumps(teacher_config.prompts.model_dump(), sort_keys=True).encode("utf-8"))
        return digest.hexdigest()


__all__ = [
    "PromptRewriteEngine",
    "RewrittenStudentPrompts",
    "RewrittenTeacherPrompts",
]

