import datetime

import pytest

from atlas.runtime.persona_memory.merge import PersonaMemoryInstruction, merge_prompt, normalize_instructions


def test_merge_prompt_with_string_and_dict_instructions():
    base_prompt = "Base instructions"
    instructions = [
        PersonaMemoryInstruction(memory_id=1, created_at=datetime.datetime(2024, 1, 1), payload="Remember to cite sources."),
        PersonaMemoryInstruction(memory_id=2, created_at=datetime.datetime(2024, 1, 2), payload={"prepend": "Act as a domain expert.", "append": "Summarize findings."}),
    ]
    merged = merge_prompt(base_prompt, instructions)
    assert "Act as a domain expert." in merged.split("\n")[0]
    assert "Remember to cite sources." in merged
    assert merged.endswith("Summarize findings.")


def test_merge_prompt_with_replace_instruction():
    base_prompt = "Original"
    instructions = [
        PersonaMemoryInstruction(memory_id=1, created_at=datetime.datetime(2024, 1, 1), payload={"replace": "New base", "append": "Extra context"}),
    ]
    merged = merge_prompt(base_prompt, instructions)
    assert merged == "New base\n\nExtra context"


def test_normalize_instructions():
    records = [
        {"memory_id": "a", "created_at": "2024-01-01", "instruction": "note"},
        {"memory_id": "b", "created_at": "2024-01-02", "instruction": {"append": "more"}},
    ]
    normalized = normalize_instructions(records)
    assert normalized[0].memory_id == "a"
    assert normalized[1].payload == {"append": "more"}
