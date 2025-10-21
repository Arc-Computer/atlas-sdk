
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import eval_probe_models as probe_eval
from scripts import export_probe_dataset as probe_export
from scripts.eval_probe_models import ProbeResult, ProbeSample, summarise_results


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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


@pytest.mark.anyio("asyncio")
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


def test_export_helpers_extract_learning_key() -> None:
    metadata = {"learning_key": "abc", "session_metadata": {"learning_key": "xyz"}}
    assert probe_export._extract_learning_key(metadata) == "abc"
    assert probe_export._extract_learning_key({"session_metadata": {"learning_key": "zzz"}}) == "zzz"
    assert probe_export._expected_mode({"adaptive_summary": {"adaptive_mode": "auto"}}) == "auto"


@pytest.mark.anyio("asyncio")
async def test_export_probe_dataset_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDatabase:
        call_counts = 0

        async def connect(self):
            return None

        async def disconnect(self):
            return None

        async def fetch_sessions(self, limit: int, offset: int):
            return [
                {
                    "id": 1,
                    "task": "Task",
                    "metadata": json.dumps(
                        {
                            "learning_key": "key-1",
                            "adaptive_summary": {"adaptive_mode": "auto"},
                        }
                    ),
                    "created_at": "2025-01-01T00:00:00Z",
                    "status": "succeeded",
                }
            ,
                {
                    "id": 2,
                    "task": "Task",
                    "metadata": json.dumps(
                        {
                            "learning_key": "key-1",
                            "adaptive_summary": {"adaptive_mode": "auto"},
                        }
                    ),
                    "created_at": "2025-01-02T00:00:00Z",
                    "status": "succeeded",
                }
            ]

        async def fetch_learning_history(self, learning_key: str):
            assert learning_key == "key-1"
            return [
                {
                    "reward": {"score": 0.9},
                    "student_learning": "note",
                    "teacher_learning": "coach",
                    "created_at": "2025-01-01T00:00:00Z",
                    "completed_at": "2025-01-01T00:10:00Z",
                }
            ]

    monkeypatch.setattr(probe_export, "Database", lambda config: FakeDatabase())

    output = tmp_path / "dataset.jsonl"
    written = await probe_export.export_probe_dataset(
        database_url="postgresql://example",  # ignored by fake
        output_path=output,
        session_limit=10,
        session_offset=0,
        history_limit=None,
        min_history=1,
        include_missing_mode=False,
        per_learning_key_limit=1,
    )

    assert written == 1
    content = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    payload = json.loads(content[0])
    assert payload["expected_mode"] == "auto"
