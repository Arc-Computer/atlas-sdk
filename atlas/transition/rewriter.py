"""Prompt rewrite engine that derives planner/teacher personas via LLM."""

from __future__ import annotations

import hashlib
import json
import re
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
                    " Student, and a Teacher reviewer.\n\n"
                    "CRITICAL: You MUST return a JSON object with this EXACT structure:\n"
                    "{\n"
                    '  "student": {\n'
                    '    "planner": "<planner prompt>",\n'
                    '    "executor": "<executor prompt>",\n'
                    '    "synthesizer": "<synthesizer prompt>"\n'
                    "  },\n"
                    '  "teacher": {\n'
                    '    "plan_review": "<plan review prompt>",\n'
                    '    "validation": "<validation prompt>",\n'
                    '    "guidance": "<guidance prompt>"\n'
                    "  }\n"
                    "}\n\n"
                    "Use ONLY these keys: 'student', 'teacher', 'planner', 'executor', 'synthesizer', "
                    "'plan_review', 'validation', 'guidance'. Do NOT use alternative key names."
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

        if not response.content or not response.content.strip():
            raise ValueError(
                f"Prompt rewrite LLM returned empty response. "
                f"Raw response object: {response.raw}"
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
                "Generate concise, fully-specified system prompts. Planner must emit JSON where each step entry includes"
                " the keys id (integer), description (string), depends_on (list of integers), tool (string or null),"
                " and tool_params (object or null) only. Steps must form a strictly sequential chain: step 1 has no"
                " dependencies, step 2 depends on [1], step 3 depends on [2], etc. No parallel execution is allowed."
                " Executor should follow plans with precise instructions for tool usage; teacher should review, validate,"
                " and provide remediation guidance. Maintain the tone and constraints of the base agent prompt."
            ),
        }

    def _extract_json(self, content: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown code fences and extra text."""
        content_stripped = content.strip()

        try:
            return json.loads(content_stripped)
        except json.JSONDecodeError:
            pass

        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        first_brace = content.find('{')
        if first_brace != -1:
            try:
                return json.loads(content[first_brace:])
            except json.JSONDecodeError:
                pass

            depth = 0
            for i in range(first_brace, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[first_brace:i+1])
                        except json.JSONDecodeError:
                            break

        return None

    def _parse_response(
        self,
        content: str,
    ) -> tuple[RewrittenStudentPrompts, RewrittenTeacherPrompts]:
        data = self._extract_json(content)
        if data is None:
            raise ValueError(
                f"Prompt rewrite LLM response was not valid JSON. "
                f"Response preview: {content[:500]}"
            )

        try:
            student = data.get("student") or data.get("student_prompts")
            teacher = data.get("teacher") or data.get("teacher_prompts")
            if not student or not teacher:
                raise KeyError("Missing student or teacher sections")
        except (KeyError, AttributeError) as exc:
            raise ValueError(
                f"Prompt rewrite JSON missing required sections. "
                f"Found keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
            ) from exc

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
