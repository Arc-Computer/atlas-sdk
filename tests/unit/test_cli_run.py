import argparse
import json

from atlas.cli.run import _run_with_config
from atlas.runtime.orchestration.execution_context import ExecutionContext


class _DummyResult:
    def __init__(self, final_answer: str = "done") -> None:
        self.final_answer = final_answer

    def model_dump(self) -> dict[str, str]:
        return {"final_answer": self.final_answer}


async def _fake_arun(task: str, config_path: str, stream_progress: bool, session_metadata: dict[str, object]):
    context = ExecutionContext.get()
    context.metadata.update(
        {
            "steps": {
                1: {
                    "attempts": [
                        {
                            "evaluation": {
                                "validation": {
                                    "cached": True,
                                    "validation_request": {
                                        "structured_output": {"content": "ok"},
                                        "artifacts": {"content": ["item"]},
                                        "deliverable": {"content": "deliverable"},
                                        "prior_results": {"content": []},
                                    },
                                }
                            }
                        }
                    ]
                }
            }
        }
    )
    return _DummyResult()


def test_run_with_config_persists_metadata(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("atlas.cli.run.atlas_arun", _fake_arun)
    monkeypatch.setattr("atlas.cli.run.load_dotenv_if_available", lambda *_, **__: None)

    args = argparse.Namespace(
        config=str(config_path),
        task="demo task",
        path=str(tmp_path),
        mode=None,
        max_steps=None,
    )

    try:
        exit_code = _run_with_config(args)
    finally:
        # Ensure future tests start with a clean execution context.
        ExecutionContext.get().reset()

    assert exit_code == 0
    _ = capsys.readouterr()

    runs_dir = tmp_path / ".atlas" / "runs"
    run_files = list(runs_dir.glob("run_*.json"))
    assert len(run_files) == 1
    payload = json.loads(run_files[0].read_text(encoding="utf-8"))

    metadata = payload.get("metadata") or {}
    # Step identifiers should be coerced to strings for JSON compatibility.
    step_entry = metadata.get("steps", {}).get("1", {})
    attempts = step_entry.get("attempts", [])
    assert attempts, "expected run metadata to include at least one attempt entry"

    validation_meta = attempts[0].get("evaluation", {}).get("validation", {})
    assert validation_meta.get("cached") is True
    assert "validation_request" in validation_meta
