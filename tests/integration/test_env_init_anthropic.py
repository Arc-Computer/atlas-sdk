"""Integration tests for atlas env init with Anthropic-only defaults and LLM-based selection."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atlas.cli.config_defaults import build_default_config, validate_anthropic_key
from atlas.cli.candidate_selection import (
    CandidateSelection,
    heuristic_rank_candidates,
    llm_select_candidate,
)
from atlas.sdk.discovery import Candidate


@pytest.fixture
def sample_candidates():
    """Create sample candidates for testing."""
    return [
        Candidate(
            file_path=Path("/tmp/agent1.py"),
            module="agent1",
            class_name="Agent1",
            score=0.8,
            via_decorator=True,
            role="agent",
            capability_methods=["step", "assess"],
        ),
        Candidate(
            file_path=Path("/tmp/agent2.py"),
            module="agent2",
            class_name="Agent2",
            score=0.6,
            via_decorator=False,
            role="agent",
            capability_methods=["step"],
        ),
    ]


class TestHeuristicRanking:
    """Test heuristic candidate ranking fallback."""

    def test_single_candidate(self, sample_candidates):
        """Test that single candidate is returned directly."""
        result = heuristic_rank_candidates([sample_candidates[0]])
        assert result == sample_candidates[0]

    def test_decorated_candidate_preferred(self, sample_candidates):
        """Test that decorated candidates are preferred."""
        result = heuristic_rank_candidates(sample_candidates)
        assert result == sample_candidates[0]  # Agent1 has decorator
        assert result.via_decorator is True

    def test_higher_score_wins_without_decorator(self):
        """Test that higher score wins when neither has decorator."""
        candidates = [
            Candidate(
                file_path=Path("/tmp/a.py"),
                module="a",
                class_name="A",
                score=0.5,
                via_decorator=False,
                role="agent",
            ),
            Candidate(
                file_path=Path("/tmp/b.py"),
                module="b",
                class_name="B",
                score=0.7,
                via_decorator=False,
                role="agent",
            ),
        ]
        result = heuristic_rank_candidates(candidates)
        assert result.class_name == "B"
        assert result.score == 0.7


class TestLLMSelection:
    """Test LLM-based candidate selection."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not available",
    )
    def test_llm_selection_with_real_api(self, sample_candidates):
        """Test LLM selection with real Anthropic API."""
        result = llm_select_candidate(sample_candidates)

        assert isinstance(result, CandidateSelection)
        assert result.candidate in sample_candidates
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning

    def test_llm_selection_single_candidate(self, sample_candidates):
        """Test that single candidate returns confidence 1.0."""
        result = llm_select_candidate([sample_candidates[0]])

        assert result.candidate == sample_candidates[0]
        assert result.confidence == 1.0
        assert "only one" in result.reasoning.lower()

    @patch("atlas.cli.candidate_selection.os.getenv")
    def test_llm_selection_no_api_key_fallback(self, mock_getenv, sample_candidates):
        """Test fallback to heuristic when API key missing."""
        mock_getenv.return_value = None

        result = llm_select_candidate(sample_candidates)

        # Should fall back to heuristic selection
        assert result.candidate == sample_candidates[0]  # Decorated candidate
        assert result.confidence == 0.7  # Heuristic confidence
        assert "heuristic" in result.reasoning.lower()

    @patch("atlas.cli.candidate_selection.anthropic.Anthropic")
    def test_llm_selection_api_error_fallback(self, mock_anthropic, sample_candidates):
        """Test fallback to heuristic when API call fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic.return_value = mock_client

        with patch("atlas.cli.candidate_selection.os.getenv", return_value="test-key"):
            result = llm_select_candidate(sample_candidates)

        # Should fall back to heuristic selection
        assert result.candidate == sample_candidates[0]
        assert "fallback" in result.reasoning.lower()


class TestAnthropicDefaults:
    """Test Anthropic-only default configuration."""

    def test_validate_anthropic_key_present(self):
        """Test key validation when present."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert validate_anthropic_key() is True

    def test_validate_anthropic_key_missing(self):
        """Test key validation when missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert validate_anthropic_key() is False

    def test_build_default_config_structure(self, sample_candidates):
        """Test that default config has correct structure."""
        config = build_default_config(
            sample_candidates[0],
            tools=[],
            storage_url="postgresql://atlas:atlas@localhost:5433/atlas",
        )

        # Verify Anthropic models
        assert config.teacher.llm.provider.value == "anthropic"
        assert config.teacher.llm.model == "claude-sonnet-4-5-20250929"
        assert config.teacher.llm.api_key_env == "ANTHROPIC_API_KEY"

        # Verify RIM uses Anthropic models
        assert config.rim.small_model.provider.value == "anthropic"
        assert config.rim.small_model.model == "claude-4-5-haiku-20241022"
        assert config.rim.large_model.provider.value == "anthropic"
        assert config.rim.large_model.model == "claude-sonnet-4-5-20250929"

        # Verify learning enabled with few-shot and playbook
        assert config.learning.enabled is True
        assert config.learning.update_enabled is True
        assert config.learning.inject_few_shot_examples is True
        assert config.learning.playbook_injection_mode == "prefix"
        assert config.learning.apply_to_prompts is True

        # Verify storage configured
        assert config.storage is not None
        assert config.storage.database_url == "postgresql://atlas:atlas@localhost:5433/atlas"

    def test_build_default_config_without_storage(self, sample_candidates):
        """Test config generation without storage."""
        config = build_default_config(sample_candidates[0], tools=[])

        assert config.storage is None


class TestStorageSetup:
    """Test storage setup integration."""

    @patch("atlas.cli.env.subprocess.run")
    def test_storage_already_running(self, mock_run):
        """Test detection of already running storage."""
        from atlas.cli.env import _setup_storage_if_needed

        # Mock docker ps showing atlas-postgres already running
        mock_run.return_value = MagicMock(
            stdout="atlas-postgres", returncode=0
        )

        result = _setup_storage_if_needed(interactive=False)

        assert result == "postgresql://atlas:atlas@localhost:5433/atlas"
        # Should only check status, not start containers
        assert mock_run.call_count == 1

    @patch("atlas.cli.env.subprocess.run")
    @patch("atlas.cli.env.input")
    def test_storage_setup_user_declines(self, mock_input, mock_run):
        """Test user declining storage setup."""
        from atlas.cli.env import _setup_storage_if_needed

        # Mock docker ps showing no atlas-postgres
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        # User declines
        mock_input.return_value = "n"

        result = _setup_storage_if_needed(interactive=True)

        assert result is None
        # Should only check status, not start containers
        assert mock_run.call_count == 1

    @patch("atlas.cli.env.subprocess.run")
    def test_storage_setup_success(self, mock_run):
        """Test successful storage setup."""
        from atlas.cli.env import _setup_storage_if_needed

        # Mock docker ps showing no atlas-postgres
        # Then mock docker compose up success
        mock_run.side_effect = [
            MagicMock(stdout="", returncode=0),  # docker ps
            MagicMock(returncode=0),  # docker compose up
        ]

        result = _setup_storage_if_needed(interactive=False)

        assert result == "postgresql://atlas:atlas@localhost:5433/atlas"
        assert mock_run.call_count == 2


class TestEndToEndEnvInit:
    """End-to-end integration tests for env init."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not available",
    )
    def test_env_init_with_simple_agent(self):
        """Test full env init flow with a simple agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple agent file
            agent_file = Path(tmpdir) / "my_agent.py"
            agent_file.write_text(
                """
from atlas.sdk import agent

@agent(name="TestAgent")
class TestAgent:
    def step(self, input: str) -> str:
        return input.upper()
"""
            )

            # Run discovery
            from atlas.sdk.discovery import discover_candidates

            candidates = discover_candidates(Path(tmpdir))
            agent_candidates = [c for c in candidates if c.role == "agent"]

            assert len(agent_candidates) > 0

            # Test LLM selection
            from atlas.cli.candidate_selection import llm_select_candidate

            selection = llm_select_candidate(agent_candidates)

            assert selection.candidate.class_name == "TestAgent"
            assert selection.confidence > 0.0

            # Test config generation
            from atlas.cli.config_defaults import build_default_config

            config = build_default_config(selection.candidate)

            assert config.agent.name == "TestAgent"
            assert config.learning.inject_few_shot_examples is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
