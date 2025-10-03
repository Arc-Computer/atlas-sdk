import asyncio
import contextlib
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from atlas.config.loader import load_config
from atlas.dashboard.server import create_dashboard_app
from atlas.orchestration.execution_context import ExecutionContext
from examples.gdpval import loader
from examples.gdpval import run_gdpval


def _build_reference(cache_root: Path, task_id: str) -> loader.GDPValReference:
    ref_dir = cache_root / task_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    cached_path = ref_dir / "evidence.txt"
    cached_path.write_text("evidence line a\nevidence line b")
    text_path = cached_path.with_suffix(".txt")
    text_path.write_text("evidence line a\nevidence line b")
    return loader.GDPValReference(
        filename="evidence.txt",
        media_type="text/plain",
        cached_path=cached_path,
        text_path=text_path,
        metadata={
            "cached_path": str(cached_path),
            "text_path": str(text_path),
            "media_type": "text/plain",
        },
    )


@pytest.mark.gdpval
def test_run_task_collects_metadata(tmp_path, monkeypatch):
    async def runner() -> None:
        cache_root = Path(tmp_path / "cache")
        monkeypatch.setattr(loader, "CACHE_ROOT", cache_root)
        monkeypatch.setattr(run_gdpval, "CACHE_ROOT", cache_root, raising=False)
        task_id = f"gdpval-{uuid.uuid4()}"
        reference = _build_reference(cache_root, task_id)
        task = loader.GDPValTask(
            task_id=task_id,
            sector="manufacturing",
            occupation="engineer",
            prompt="Evaluate productivity.",
            references=[reference],
        )
        publisher = run_gdpval.TelemetryPublisher()
        events: list[dict] = []

        async def consume() -> None:
            async for payload in publisher.stream():
                events.append(payload)

        collector = asyncio.create_task(consume())
        config_path = "configs/examples/gdpval_python.yaml"
        try:
            record = await run_gdpval._run_task(task=task, config_path=config_path, publisher=publisher)
        except Exception:
            snapshot = ExecutionContext.get().metadata
            print(f"Execution metadata snapshot: {snapshot}")
            raise
        finally:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
        metadata = ExecutionContext.get().metadata
        assert metadata.get("session_metadata", {}).get("task_id") == task_id
        assert "prompt_rewrite" in metadata
        telemetry_types = {item.get("type") for item in events}
        assert "intermediate-step" in telemetry_types
        assert record["task_id"] == task_id
        assert record["final_answer"].strip()
        manifest_path = cache_root / task_id / "manifest.json"
        assert manifest_path.exists()
        atlas_config = load_config(config_path)
        app = create_dashboard_app(database_url=atlas_config.storage.database_url)
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/sessions")
            response.raise_for_status()
            sessions = response.json()["sessions"]
            matching = [
                entry
                for entry in sessions
                if entry.get("metadata", {}).get("session_metadata", {}).get("task_id") == task_id
            ]
            if not matching:
                pytest.fail("GDPval session not present in dashboard API")

    asyncio.run(runner())
