from pathlib import Path

import pytest

from examples.gdpval import loader


class FakeDataset(list):
    pass


def test_load_tasks_caches_references(tmp_path, monkeypatch):
    fake_data = FakeDataset(
        [
            {
                "task_id": "task-1",
                "sector": "energy",
                "occupation": "analyst",
                "prompt": "Assess the energy outlook.",
                "reference_files": ["sample.txt"],
                "reference_file_urls": ["https://example.com/sample.txt"],
                "reference_file_hf_uris": ["hf://datasets/openai/gdpval/resolve/train/sample.txt"],
                "reference_media_types": ["text/plain"],
            }
        ]
    )

    def fake_dataset(*_args, **_kwargs):
        return fake_data

    def fake_fetch(_url: str | None, _hf_uri: str | None) -> bytes:
        return b"GDPval reference body"

    monkeypatch.setattr(loader, "CACHE_ROOT", Path(tmp_path / "cache"))
    monkeypatch.setattr(
        loader,
        "datasets",
        type("Stub", (), {"load_dataset": staticmethod(fake_dataset)})(),
    )
    monkeypatch.setattr(loader, "_DATASETS_ERROR", None)

    tasks = list(
        loader.load_gdpval_tasks(
            split="train",
            streaming=False,
            cache_references=True,
            fetch=fake_fetch,
        )
    )

    assert len(tasks) == 1
    manifest_path = loader.CACHE_ROOT / "task-1" / "manifest.json"
    assert manifest_path.exists()
    manifest = loader.load_manifest("task-1")
    reference = manifest["references"][0]
    cached_path = Path(reference["cached_path"])
    text_path = Path(reference["text_path"])
    assert cached_path.read_bytes() == b"GDPval reference body"
    assert "GDPval reference" in text_path.read_text()
    assert reference["source_hf_uri"].startswith("hf://")


def test_load_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CACHE_ROOT", Path(tmp_path / "cache"))
    with pytest.raises(loader.ReferenceNotFound):
        loader.load_manifest("unknown-task")
