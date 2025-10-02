"""Student persona prompt rewriting with memoisation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from atlas.config.models import StudentPrompts


@dataclass(frozen=True)
class RewrittenPrompts:
    planner: str
    executor: str
    synthesizer: str


class PromptRewriter:
    def __init__(self) -> None:
        self._cache: dict[str, RewrittenPrompts] = {}

    def rewrite(self, base_prompt: str, prompts: StudentPrompts) -> RewrittenPrompts:
        cache_key = self._cache_key(base_prompt, prompts)
        if cache_key in self._cache:
            return self._cache[cache_key]
        rewritten = RewrittenPrompts(
            planner=self._inject(base_prompt, prompts.planner),
            executor=self._inject(base_prompt, prompts.executor),
            synthesizer=self._inject(base_prompt, prompts.synthesizer),
        )
        self._cache[cache_key] = rewritten
        return rewritten

    def _inject(self, base_prompt: str, template: str) -> str:
        text = template.strip()
        if "{base_prompt}" in text:
            return text.replace("{base_prompt}", base_prompt.strip())
        joined = "\n\n".join(part for part in [text, base_prompt.strip()] if part)
        return joined.strip()

    def _cache_key(self, base_prompt: str, prompts: StudentPrompts) -> str:
        digest = hashlib.sha256()
        digest.update(base_prompt.strip().encode("utf-8"))
        digest.update(prompts.planner.strip().encode("utf-8"))
        digest.update(prompts.executor.strip().encode("utf-8"))
        digest.update(prompts.synthesizer.strip().encode("utf-8"))
        return digest.hexdigest()


__all__ = ["PromptRewriter", "RewrittenPrompts"]
