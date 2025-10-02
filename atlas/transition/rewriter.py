"""Student persona prompt rewriting with memoisation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from atlas.config.models import StudentPrompts, TeacherPrompts


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


class PromptRewriter:
    def __init__(self) -> None:
        self._student_cache: dict[str, RewrittenStudentPrompts] = {}
        self._teacher_cache: dict[str, RewrittenTeacherPrompts] = {}

    def rewrite_student(self, base_prompt: str, prompts: StudentPrompts) -> RewrittenStudentPrompts:
        cache_key = self._student_cache_key(base_prompt, prompts)
        if cache_key in self._student_cache:
            return self._student_cache[cache_key]
        rewritten = RewrittenStudentPrompts(
            planner=self._inject(base_prompt, prompts.planner),
            executor=self._inject(base_prompt, prompts.executor),
            synthesizer=self._inject(base_prompt, prompts.synthesizer),
        )
        self._student_cache[cache_key] = rewritten
        return rewritten

    def rewrite_teacher(self, base_prompt: str, prompts: TeacherPrompts) -> RewrittenTeacherPrompts:
        cache_key = self._teacher_cache_key(base_prompt, prompts)
        if cache_key in self._teacher_cache:
            return self._teacher_cache[cache_key]
        rewritten = RewrittenTeacherPrompts(
            plan_review=self._inject(base_prompt, prompts.plan_review),
            validation=self._inject(base_prompt, prompts.validation),
            guidance=self._inject(base_prompt, prompts.guidance),
        )
        self._teacher_cache[cache_key] = rewritten
        return rewritten

    # Backwards compatibility for existing callers
    def rewrite(self, base_prompt: str, prompts: StudentPrompts) -> RewrittenStudentPrompts:  # pragma: no cover
        return self.rewrite_student(base_prompt, prompts)

    def _inject(self, base_prompt: str, template: str) -> str:
        text = template.strip()
        if "{base_prompt}" in text:
            return text.replace("{base_prompt}", base_prompt.strip())
        joined = "\n\n".join(part for part in [text, base_prompt.strip()] if part)
        return joined.strip()

    def _student_cache_key(self, base_prompt: str, prompts: StudentPrompts) -> str:
        digest = hashlib.sha256()
        digest.update(base_prompt.strip().encode("utf-8"))
        digest.update(prompts.planner.strip().encode("utf-8"))
        digest.update(prompts.executor.strip().encode("utf-8"))
        digest.update(prompts.synthesizer.strip().encode("utf-8"))
        return digest.hexdigest()

    def _teacher_cache_key(self, base_prompt: str, prompts: TeacherPrompts) -> str:
        digest = hashlib.sha256()
        digest.update(base_prompt.strip().encode("utf-8"))
        digest.update(prompts.plan_review.strip().encode("utf-8"))
        digest.update(prompts.validation.strip().encode("utf-8"))
        digest.update(prompts.guidance.strip().encode("utf-8"))
        return digest.hexdigest()


__all__ = [
    "PromptRewriter",
    "RewrittenStudentPrompts",
    "RewrittenTeacherPrompts",
]
