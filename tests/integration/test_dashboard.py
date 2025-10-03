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
    cached_path.write_text("evidence data")
    text_path = cached_path.with_suffix(".txt")
    text_path.write_text("evidence data")
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


@pytest.mark.postgres
def test_dashboard_endpoints_with_postgres(tmp_path, monkeypatch):
    async def runner():
        cache_root = Path(tmp_path / "cache")
        monkeypatch.setattr(loader, "CACHE_ROOT", cache_root)
        monkeypatch.setattr(run_gdpval, "CACHE_ROOT", cache_root, raising=False)
        task_id = f"dashboard-{uuid.uuid4()}"
        reference = _build_reference(cache_root, task_id)
        task = loader.GDPValTask(
            task_id=task_id,
            sector="finance",
            occupation="analyst",
            prompt="Summarize indicators.",
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
                pytest.fail("Session not found after live run")
            session_id = matching[0]["id"]
            detail = await client.get(f"/api/sessions/{session_id}")
            detail.raise_for_status()
            session = detail.json()["session"]
            assert session["final_answer"] == record["final_answer"]
            assert session["metadata"]["session_metadata"]["task_id"] == task_id
            assert session["plan"]["steps"]
            steps_response = await client.get(f"/api/sessions/{session_id}/steps")
            steps_response.raise_for_status()
            steps = steps_response.json()["steps"]
            assert len(steps) == record["attempts"]
            assert all(step["trace"] for step in steps)
            events_response = await client.get(f"/api/sessions/{session_id}/events")
            events_response.raise_for_status()
            dashboard_events = events_response.json()["events"]
            if not dashboard_events:
                pytest.fail("No dashboard events recorded")
        assert any(item.get("type") == "intermediate-step" for item in events)

    asyncio.run(runner())
