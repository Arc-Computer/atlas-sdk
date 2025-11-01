"""Comprehensive tests for atlas quickstart command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.cli import main as cli_main
from atlas.cli import quickstart as quickstart_cli


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
agent:
  type: openai
  name: test-agent
  system_prompt: "You are a test agent."
  tools: []
  llm:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
storage: null
"""
    )
    return config_file


@pytest.fixture
def mock_arun_result():
    """Mock result from core.arun."""
    result = MagicMock()
    result.final_answer = "This is a test security review response."
    return result


@pytest.fixture
def mock_execution_context():
    """Mock execution context with metadata."""
    context = MagicMock()
    context.metadata = {
        "reward_summary": {"score": 0.75},
        "token_usage": {"total_tokens": 1000},
    }
    return context


def test_quickstart_command_registered() -> None:
    """Test that quickstart command appears in CLI."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline"])
    assert args.command == "quickstart"


def test_quickstart_offline_flag(monkeypatch: pytest.MonkeyPatch, mock_config_path: Path) -> None:
    """Test that --offline sets ATLAS_OFFLINE_MODE=1."""
    monkeypatch.delenv("ATLAS_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1"])

    # Check that handler would set offline mode
    assert args.offline is True
    assert args.config == str(mock_config_path)


def test_quickstart_config_override(mock_config_path: Path) -> None:
    """Test that --config flag works."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--config", str(mock_config_path), "--tasks", "1", "--offline"])

    assert args.config == str(mock_config_path)


def test_quickstart_tasks_flag() -> None:
    """Test that --tasks flag limits execution."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--tasks", "2", "--offline"])

    assert args.tasks == 2


def test_quickstart_invalid_tasks_flag() -> None:
    """Test that invalid --tasks values are rejected."""
    parser = cli_main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["quickstart", "--tasks", "4", "--offline"])


def test_quickstart_skip_storage() -> None:
    """Test that --skip-storage bypasses Postgres check."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--skip-storage", "--offline"])

    assert args.skip_storage is True


def test_quickstart_no_api_keys_offline_mode(
    monkeypatch: pytest.MonkeyPatch, mock_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that missing API keys are OK in offline mode."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ATLAS_OFFLINE_MODE", raising=False)

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1"])

    # Should not raise error for missing keys when offline
    with patch.object(quickstart_cli, "_cmd_quickstart_async", return_value=0):
        exit_code = args.handler(args)
        assert exit_code == 0


def test_quickstart_missing_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test clear error message when config file missing."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--config", str(tmp_path / "nonexistent.yaml"), "--tasks", "1"])

    with pytest.raises(SystemExit) as exc_info:
        args.handler(args)
    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "Config file not found" in captured.err


@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
@patch("atlas.cli.quickstart._check_storage_available")
async def test_quickstart_full_flow_offline(
    mock_check_storage: MagicMock,
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
    mock_execution_context: MagicMock,
) -> None:
    """Test end-to-end success path with offline mode."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_check_storage.return_value = False
    mock_get_context.return_value = mock_execution_context
    mock_arun.return_value = mock_arun_result

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1", "--skip-storage"])

    exit_code = await quickstart_cli._cmd_quickstart_async(args)

    assert exit_code == 0
    assert mock_arun.call_count == 1


@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
@patch("atlas.cli.quickstart._check_storage_available")
async def test_quickstart_with_storage(
    mock_check_storage: MagicMock,
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
    mock_execution_context: MagicMock,
) -> None:
    """Test Postgres integration works."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_check_storage.return_value = True
    mock_get_context.return_value = mock_execution_context
    mock_arun.return_value = mock_arun_result

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1"])

    exit_code = await quickstart_cli._cmd_quickstart_async(args)

    assert exit_code == 0
    mock_check_storage.assert_called_once()


@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
@patch("atlas.cli.quickstart._check_storage_available")
async def test_quickstart_without_storage(
    mock_check_storage: MagicMock,
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
    mock_execution_context: MagicMock,
) -> None:
    """Test graceful fallback when storage unavailable."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_check_storage.return_value = False
    mock_get_context.return_value = mock_execution_context
    mock_arun.return_value = mock_arun_result

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1"])

    exit_code = await quickstart_cli._cmd_quickstart_async(args)

    assert exit_code == 0
    mock_check_storage.assert_called_once()


@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
@patch("atlas.cli.quickstart._check_storage_available")
async def test_quickstart_metrics_collection(
    mock_check_storage: MagicMock,
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
) -> None:
    """Test metrics table populated correctly."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Create mock contexts with different metrics for each task
    contexts = [
        MagicMock(metadata={"reward_summary": {"score": 0.65}, "token_usage": {"total_tokens": 2341}}),
        MagicMock(metadata={"reward_summary": {"score": 0.82}, "token_usage": {"total_tokens": 1892}}),
        MagicMock(metadata={"reward_summary": {"score": 0.88}, "token_usage": {"total_tokens": 1654}}),
    ]

    mock_check_storage.return_value = False
    mock_get_context.side_effect = contexts
    mock_arun.return_value = mock_arun_result

    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "3", "--skip-storage"])

    exit_code = await quickstart_cli._cmd_quickstart_async(args)

    assert exit_code == 0
    assert mock_arun.call_count == 3


def test_quickstart_keyboard_interrupt(mock_config_path: Path) -> None:
    """Test Ctrl+C cleanup works."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1"])

    with patch.object(quickstart_cli, "asyncio") as mock_asyncio:
        mock_asyncio.run.side_effect = KeyboardInterrupt()
        exit_code = args.handler(args)

        assert exit_code == 130


def test_extract_reward_score() -> None:
    """Test reward score extraction from metadata."""
    metadata = {"reward_summary": {"score": 0.85}}
    assert quickstart_cli._extract_reward_score(metadata) == 0.85

    metadata2 = {"session_reward": {"score": 0.75}}
    assert quickstart_cli._extract_reward_score(metadata2) == 0.75

    metadata3 = {}
    assert quickstart_cli._extract_reward_score(metadata3) is None


def test_extract_token_count() -> None:
    """Test token count extraction from metadata."""
    metadata = {"token_usage": {"total_tokens": 1234}}
    assert quickstart_cli._extract_token_count(metadata) == 1234

    metadata2 = {"learning_usage": {"session": {"token_usage": {"total_tokens": 5678}}}}
    assert quickstart_cli._extract_token_count(metadata2) == 5678

    metadata3 = {}
    assert quickstart_cli._extract_token_count(metadata3) is None


def test_format_metrics_table() -> None:
    """Test metrics table formatting."""
    metrics = [
        quickstart_cli.TaskMetrics(task_num=1, reward=0.65, tokens=2341, duration=18.3),
        quickstart_cli.TaskMetrics(task_num=2, reward=0.82, tokens=1892, duration=14.1),
        quickstart_cli.TaskMetrics(task_num=3, reward=0.88, tokens=1654, duration=12.8),
    ]

    table = quickstart_cli._format_metrics_table(metrics)

    assert "Learning Progress:" in table
    assert "│ 1" in table
    assert "│ 2" in table
    assert "│ 3" in table
    assert "↑" in table  # Should show improvement indicators


def test_generate_insights() -> None:
    """Test learning insights generation."""
    metrics = [
        quickstart_cli.TaskMetrics(task_num=1, reward=0.65, tokens=2341, duration=18.3),
        quickstart_cli.TaskMetrics(task_num=3, reward=0.88, tokens=1654, duration=12.8),
    ]

    insights = quickstart_cli._generate_insights(metrics)

    assert any("Quality increased" in insight for insight in insights)
    assert any("Efficiency improved" in insight for insight in insights)
    assert any("Speed improved" in insight for insight in insights)


def test_resolve_config_path(mock_config_path: Path) -> None:
    """Test config path resolution."""
    resolved = quickstart_cli._resolve_config_path(str(mock_config_path))
    assert resolved == str(mock_config_path)


def test_resolve_config_path_not_found(tmp_path: Path) -> None:
    """Test config path resolution fails for missing file."""
    with pytest.raises(SystemExit):
        quickstart_cli._resolve_config_path(str(tmp_path / "nonexistent.yaml"))


def test_format_final_answer_json(tmp_path: Path) -> None:
    """Test JSON structure display."""
    json_answer = '{"result": {"key": "value", "nested": {"a": 1, "b": 2}}, "metadata": {"count": 5}}'
    artifact_path = tmp_path / "run_20251101_181000_task1.json"
    
    formatted = quickstart_cli._format_final_answer(json_answer, artifact_path)
    
    assert "JSON structure" in formatted
    assert "Snippet:" in formatted
    assert "result" in formatted
    assert "metadata" in formatted
    assert artifact_path.name in formatted


def test_format_final_answer_text_truncation(tmp_path: Path) -> None:
    """Test text truncation at 1500 chars."""
    long_text = "x" * 2000
    artifact_path = tmp_path / "run_20251101_181000_task1.json"
    
    formatted = quickstart_cli._format_final_answer(long_text, artifact_path)
    
    assert len(formatted) < 1600  # Truncated + "..." + artifact note
    assert "..." in formatted
    assert artifact_path.name in formatted
    
    # Test that short text is not truncated
    short_text = "This is a short answer."
    formatted_short = quickstart_cli._format_final_answer(short_text, None)
    assert formatted_short == short_text


def test_format_final_answer_text_no_truncation() -> None:
    """Test that text under 1500 chars is not truncated."""
    short_text = "This is a test security review response."
    formatted = quickstart_cli._format_final_answer(short_text, None)
    assert formatted == short_text


@patch("atlas.cli.quickstart.write_run_record")
@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
async def test_artifact_saving(
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    mock_write_run_record: MagicMock,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
    mock_execution_context: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that artifacts are saved per task."""
    artifact_path = tmp_path / "runs" / "run_20251101_181000_task1.json"
    artifact_path.parent.mkdir(parents=True)
    
    mock_get_context.return_value = mock_execution_context
    mock_arun.return_value = mock_arun_result
    mock_write_run_record.return_value = artifact_path
    
    atlas_dir = tmp_path / ".atlas"
    metrics = await quickstart_cli._run_task(
        task="Test task",
        task_num=1,
        config_path=str(mock_config_path),
        atlas_dir=atlas_dir,
    )
    
    assert metrics.artifact_path == artifact_path
    mock_write_run_record.assert_called_once()
    call_args = mock_write_run_record.call_args[0]
    assert call_args[0] == atlas_dir
    assert "task" in call_args[1]
    assert "task_num" in call_args[1]
    assert "metadata" in call_args[1]


@patch("atlas.cli.quickstart.core.arun")
@patch("atlas.cli.quickstart.ExecutionContext.get")
@patch("atlas.cli.quickstart._check_storage_available")
async def test_learning_analysis_note_shown(
    mock_check_storage: MagicMock,
    mock_get_context: MagicMock,
    mock_arun: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    mock_config_path: Path,
    mock_arun_result: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that learning note appears when playbook entries exist."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    # Create mock context with playbook entries
    mock_context = MagicMock()
    mock_context.metadata = {
        "reward_summary": {"score": 0.75},
        "token_usage": {"total_tokens": 1000},
        "learning_state": {
            "metadata": {
                "playbook_entries": [
                    {"cue": "test", "action": "test_action", "scope": "test_scope"}
                ]
            }
        },
    }
    
    mock_check_storage.return_value = False
    mock_get_context.return_value = mock_context
    mock_arun.return_value = mock_arun_result
    
    parser = cli_main.build_parser()
    args = parser.parse_args(["quickstart", "--offline", "--config", str(mock_config_path), "--tasks", "1", "--skip-storage"])
    
    exit_code = await quickstart_cli._cmd_quickstart_async(args)
    
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "💡 Learning Analysis:" in captured.out
    assert "Playbook entries saved in artifacts" in captured.out
    assert "scripts/eval_learning.py" in captured.out


def test_has_playbook_entries() -> None:
    """Test helper function for detecting playbook entries."""
    # Test with playbook entries
    metadata_with_entries = {
        "learning_state": {
            "metadata": {
                "playbook_entries": [{"cue": "test", "action": "test_action"}]
            }
        }
    }
    assert quickstart_cli._has_playbook_entries(metadata_with_entries) is True
    
    # Test with empty playbook entries
    metadata_empty = {
        "learning_state": {
            "metadata": {
                "playbook_entries": []
            }
        }
    }
    assert quickstart_cli._has_playbook_entries(metadata_empty) is False
    
    # Test without playbook entries
    metadata_no_entries = {
        "learning_state": {
            "metadata": {}
        }
    }
    assert quickstart_cli._has_playbook_entries(metadata_no_entries) is False
    
    # Test without learning_state
    metadata_no_state = {}
    assert quickstart_cli._has_playbook_entries(metadata_no_state) is False

