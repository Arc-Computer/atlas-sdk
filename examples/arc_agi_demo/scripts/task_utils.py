"""Shared utilities for ARC demo scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _ROOT / "configs" / "arc_agi_demo.yaml"
DEFAULT_TASK_PATH = _ROOT / "data" / "arc_training_0ca9ddb6.json"
_TRAINING_BASE_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training"


def format_grid(grid: Iterable[Iterable[int]]) -> str:
    rows: List[str] = []
    for row in grid:
        rows.append(" ".join(str(cell) for cell in row))
    return "\n".join(rows)


def arc_task_to_prompt(payload: Dict[str, Any]) -> str:
    sections: List[str] = []
    for index, pair in enumerate(payload.get("train", []), start=1):
        sections.append(
            "\n".join(
                [
                    f"Example {index} Input:",
                    format_grid(pair["input"]),
                    f"Example {index} Output:",
                    format_grid(pair["output"]),
                ]
            )
        )
    for index, pair in enumerate(payload.get("test", []), start=1):
        sections.append(
            "\n".join(
                [
                    f"Test Input {index}:",
                    format_grid(pair["input"]),
                    "Produce the corresponding output grid and return JSON rows.",
                ]
            )
        )
    header = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.",
        "Use the examples to infer the transformation before answering.",
    ]
    return "\n\n".join(header + sections)


def load_arc_task(task_id: str | None, task_path: Path | None) -> Dict[str, Any]:
    if task_path:
        resolved = task_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"ARC task file not found: {resolved}")
        return json.loads(resolved.read_text(encoding="utf-8"))
    if not task_id:
        default_path = DEFAULT_TASK_PATH
        if not default_path.exists():
            raise FileNotFoundError("Default ARC task missing; provide --task-file or --task-id.")
        return json.loads(default_path.read_text(encoding="utf-8"))
    url = f"{_TRAINING_BASE_URL}/{task_id}.json"
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()

