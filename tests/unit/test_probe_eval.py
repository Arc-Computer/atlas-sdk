from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scripts import eval_probe_models as probe_eval
from scripts.eval_probe_models import ProbeResult, ProbeSample, summarise_results


def test_load_dataset_sample() -> None:
    path = Path("atlas/data/sample_probe_payloads.jsonl")
    samples = probe_eval.load_dataset(path)
    assert samples, "sample dataset should contain entries"
    sample = samples[0]
    assert isinstance(sample.task, str)
    assert "entries" in sample.learning_history


def test_build_parameters_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_PROBE_MODEL_GEMINI", "gemini/custom-test-model")
    params = probe_eval.build_parameters("gemini")
    assert params.model == "gemini/custom-test-model"
    assert params.provider.value == "google"


@pytest.mark.asyncio
async def test_evaluate_model_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun(self, *, task, dossier, execution_metadata):
        class Decision:
            mode = "auto"
            confidence = 0.9

        return Decision()

    monkeypatch.setattr(probe_eval.CapabilityProbeClient, "arun", fake_arun)
    samples = [
        ProbeSample(
            task="Task A",
            learning_history={"entries": []},
            expected_mode="auto",
            metadata={},
        )
    ]
    results = await probe_eval.evaluate_model("gemini", samples, repeats=1)
    assert results[0].mode == "auto"
    assert results[0].error is None


def test_summarise_results_accuracy() -> None:
    sample = ProbeSample(
        task="Task",
        learning_history={},
        expected_mode="auto",
        metadata={},
    )
    results = [
        ProbeResult(sample=sample, model="gemini", mode="auto", confidence=0.8, latency=0.1, error=None),
        ProbeResult(sample=sample, model="gemini", mode="paired", confidence=0.4, latency=0.2, error=None),
    ]
    summary = summarise_results(results)
    assert summary["total"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
