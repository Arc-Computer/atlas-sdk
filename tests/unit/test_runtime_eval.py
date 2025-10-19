from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.config.loader import load_config
from scripts import eval_dual_agent_models as eval_mod


def test_load_dataset_parses_entries(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    lines = [
        "# comment explaining dataset shape",
        json.dumps(
            {
                "task": "Investigate alert",
                "expected_answer": "Roll back release and monitor.",
                "metadata": {"scenario": "sre", "difficulty": "medium"},
            }
        ),
    ]
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tasks = eval_mod.load_dataset(dataset_path)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.task == "Investigate alert"
    assert task.expected_answer == "Roll back release and monitor."
    assert task.metadata == {"scenario": "sre", "difficulty": "medium"}


def test_override_config_sets_llms(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("ATLAS_MODEL_OVERRIDE_") or key == "ATLAS_MODEL_TIMEOUT":
            monkeypatch.delenv(key, raising=False)

    base_config = load_config("configs/examples/openai_agent.yaml")
    overridden = eval_mod.override_config(
        base_config,
        student_model="gpt-5-mini",
        teacher_model="claude-sonnet-4-5-20250929",
    )

    student_preset = eval_mod.MODEL_PRESETS["gpt-5-mini"]
    teacher_preset = eval_mod.MODEL_PRESETS["claude-sonnet-4-5-20250929"]

    assert overridden.agent.llm.model == student_preset["model"]
    assert overridden.agent.llm.provider == student_preset["provider"]
    assert overridden.teacher.llm.model == teacher_preset["model"]
    assert overridden.teacher.llm.provider == teacher_preset["provider"]
    assert overridden.teacher.llm.temperature <= 0.15


def test_override_config_openai_adapter_enforces_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = load_config("configs/examples/openai_agent.yaml")
    overridden = eval_mod.override_config(
        base_config,
        student_model="claude-haiku-4-5",
        teacher_model="gpt-5",
    )

    assert overridden.agent.llm.provider == eval_mod.LLMProvider.OPENAI


def test_run_evaluations_with_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            (
                "# sample dataset",
                json.dumps(
                    {
                        "task": "Provide a sanitized incident summary for the finance audit channel.",
                        "expected_answer": "Share sanitized summary with controls and next steps.",
                        "metadata": {"scenario": "compliance", "difficulty": "low"},
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_execute_task(task, config_path):
        metadata = {
            "adaptive_summary": {
                "adaptive_mode": "auto",
                "mode_history": [{"mode": "auto", "confidence": 0.92}],
            },
            "session_reward": {"score": 0.8},
        }
        return SimpleNamespace(final_answer="Share sanitized summary with controls and next steps."), metadata

    monkeypatch.setattr(eval_mod, "execute_task", fake_execute_task)

    args = argparse.Namespace(
        dataset=dataset_path,
        base_config=Path("configs/examples/openai_agent.yaml"),
        student_models=["gpt-5-mini"],
        teacher_models=["gpt-5"],
        repeats=1,
        concurrency=1,
        output=None,
        similarity_threshold=0.5,
    )

    task_results, summaries = eval_mod.run_evaluations(args)

    assert len(task_results) == 1
    record = task_results[0]
    assert record.success is True
    assert record.matches is True
    assert record.similarity and record.similarity >= 0.5
    assert record.session_reward == pytest.approx(0.8)
    assert record.adaptive_mode == "auto"
    assert record.adaptive_mode_history == [{"mode": "auto", "confidence": 0.92}]

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["failures"] == 0
    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["adaptive_modes"].get("auto") == 1
