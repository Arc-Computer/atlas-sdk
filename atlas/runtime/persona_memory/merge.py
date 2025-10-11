"""Helpers to merge persona memories into base prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class PersonaMemoryInstruction:
    """Normalized persona memory instruction payload."""

    memory_id: Any
    created_at: Any
    payload: Any
    status: str | None = None


def _render_instruction(base_prompt: str, instruction: Any) -> str:
    if instruction is None:
        return base_prompt
    if isinstance(instruction, str):
        return f"{base_prompt.strip()}\n\n--- Persona Memory ---\n{instruction.strip()}"
    if isinstance(instruction, dict):
        prepend = instruction.get("prepend")
        append = instruction.get("append")
        replace = instruction.get("replace")
        parts: list[str] = []
        if isinstance(prepend, str) and prepend.strip():
            parts.append(prepend.strip())
        if replace is not None:
            if isinstance(replace, str):
                parts.append(replace.strip())
            elif isinstance(replace, dict):
                parts.append(_render_instruction("", replace))
            else:
                parts.append(str(replace))
            if isinstance(append, str) and append.strip():
                parts.append(append.strip())
            return "\n\n".join([part for part in parts if part])
        base_clean = base_prompt.strip()
        if base_clean:
            parts.append(base_clean)
        elif base_prompt and not parts:
            parts.append(base_prompt)
        if isinstance(append, str) and append.strip():
            parts.append(append.strip())
        return "\n\n".join([part for part in parts if part])
    return f"{base_prompt.strip()}\n\n--- Persona Memory ---\n{str(instruction).strip()}"


def merge_prompt(base_prompt: str, instructions: Sequence[PersonaMemoryInstruction]) -> str:
    """Merge a base prompt with persona memories ordered by creation time."""
    if not instructions:
        return base_prompt
    merged = base_prompt
    ordered = sorted(instructions, key=lambda inst: inst.created_at or "")
    for inst in ordered:
        merged = _render_instruction(merged, inst.payload)
    return merged


def normalize_instructions(records: Iterable[Dict[str, Any]]) -> List[PersonaMemoryInstruction]:
    """Convert persona memory rows into normalized instructions."""
    result: List[PersonaMemoryInstruction] = []
    for record in records:
        result.append(
            PersonaMemoryInstruction(
                memory_id=record.get("memory_id"),
                created_at=record.get("created_at"),
                payload=record.get("instruction"),
                status=record.get("status"),
            )
        )
    return result
