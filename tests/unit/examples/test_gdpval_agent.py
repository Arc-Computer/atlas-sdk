from pathlib import Path

import pytest

from examples.gdpval import agent
from examples.gdpval import loader


def test_toolset_lists_and_reads_references(tmp_path, monkeypatch):
    cache_root = Path(tmp_path / "cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = cache_root / "task-5"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    text_path = manifest_dir / "reference.txt"
    text_path.write_text("line1\nline2\nline3")
    manifest_dir.joinpath("manifest.json").write_text(
        """
        {{
          "task_id": "task-5",
          "sector": "finance",
          "occupation": "analyst",
          "prompt": "Describe trends",
          "references": [
            {{
              "filename": "reference.txt",
              "cached_path": "{cached}",
              "text_path": "{text}",
              "source_url": "https://example.com/ref.txt"
            }}
          ]
        }}
        """.format(cached=text_path, text=text_path)
    )

    monkeypatch.setattr(agent, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(loader, "CACHE_ROOT", cache_root)

    toolset = agent.GDPValToolset(cache_root=cache_root)
    references = toolset.list_references("task-5")
    assert references[0]["filename"] == "reference.txt"
    assert "line1" in toolset.read_reference("task-5", "reference.txt")
    summary = toolset.summarize_reference("task-5", "reference.txt", max_lines=2)
    assert summary.endswith("...")


def test_read_reference_missing(tmp_path, monkeypatch):
    cache_root = Path(tmp_path / "cache")
    monkeypatch.setattr(agent, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(loader, "CACHE_ROOT", cache_root)
    toolset = agent.GDPValToolset(cache_root=cache_root)
    with pytest.raises(loader.ReferenceNotFound):
        toolset.read_reference("task-x", "missing.txt")


def test_build_session_metadata():
    task = loader.GDPValTask(
        task_id="123",
        sector="energy",
        occupation="analyst",
        prompt="Explain",
        references=[],
    )
    metadata = agent.build_session_metadata(task)
    assert metadata["task_id"] == "123"
    assert metadata["sector"] == "energy"
