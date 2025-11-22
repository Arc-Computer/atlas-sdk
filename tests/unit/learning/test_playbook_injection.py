"""Unit tests for playbook injection modes and few-shot examples."""

import pytest
from unittest.mock import MagicMock, patch

from atlas.learning.playbook import extract_few_shot_examples, _redact_sensitive_data
from atlas.runtime.orchestration.execution_context import ExecutionContext


class TestPlaybookInjectionModes:
    """Test prefix and suffix injection modes."""

    def test_prefix_injection_student(self):
        """Verify that prefix mode places playbook before base prompt in Student."""
        from atlas.personas.student import Student
        from atlas.config.models import AtlasConfig, AdapterConfig, StudentConfig

        # Create minimal config
        config = AtlasConfig(
            agent=AdapterConfig(
                type="openai",
                name="test",
                system_prompt="Base prompt",
            ),
            student=StudentConfig(),
        )

        # Mock ExecutionContext with prefix mode
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_injection_mode": "prefix",
                "learning_inject_few_shot": False,
                "learning_state": {
                    "student_learning": "Student playbook content",
                },
            }
            mock_ctx.return_value = mock_instance

            student = Student(config)
            result = student._compose_system_prompt("Base prompt", "Student Playbook")

            # Playbook should come BEFORE base prompt
            assert "Student Playbook" in result
            assert "Base prompt" in result
            assert result.index("Student playbook content") < result.index("Base prompt")

    def test_suffix_injection_student(self):
        """Verify that suffix mode places playbook after base prompt in Student."""
        from atlas.personas.student import Student
        from atlas.config.models import AtlasConfig, AdapterConfig, StudentConfig

        config = AtlasConfig(
            agent=AdapterConfig(
                type="openai",
                name="test",
                system_prompt="Base prompt",
            ),
            student=StudentConfig(),
        )

        # Mock ExecutionContext with suffix mode
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_injection_mode": "suffix",
                "learning_inject_few_shot": False,
                "learning_state": {
                    "student_learning": "Student playbook content",
                },
            }
            mock_ctx.return_value = mock_instance

            student = Student(config)
            result = student._compose_system_prompt("Base prompt", "Student Playbook")

            # Playbook should come AFTER base prompt
            assert "Student Playbook" in result
            assert "Base prompt" in result
            assert result.index("Base prompt") < result.index("Student playbook content")

    def test_prefix_injection_teacher(self):
        """Verify that prefix mode places playbook before base prompt in Teacher."""
        from atlas.personas.teacher import Teacher
        from atlas.config.models import AtlasConfig, TeacherConfig

        config = AtlasConfig(
            teacher=TeacherConfig(),
        )

        # Mock ExecutionContext with prefix mode
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_injection_mode": "prefix",
                "learning_inject_few_shot": False,
                "learning_state": {
                    "teacher_learning": "Teacher playbook content",
                },
            }
            mock_ctx.return_value = mock_instance

            teacher = Teacher(config)
            result = teacher._decorate_prompt("Base prompt", "Teacher playbook content", {})

            # Playbook should come BEFORE base prompt
            assert "Teacher Playbook" in result
            assert "Base prompt" in result
            assert result.index("Teacher playbook content") < result.index("Base prompt")

    def test_suffix_injection_teacher(self):
        """Verify that suffix mode places playbook after base prompt in Teacher."""
        from atlas.personas.teacher import Teacher
        from atlas.config.models import AtlasConfig, TeacherConfig

        config = AtlasConfig(
            teacher=TeacherConfig(),
        )

        # Mock ExecutionContext with suffix mode
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_injection_mode": "suffix",
                "learning_inject_few_shot": False,
                "learning_state": {
                    "teacher_learning": "Teacher playbook content",
                },
            }
            mock_ctx.return_value = mock_instance

            teacher = Teacher(config)
            result = teacher._decorate_prompt("Base prompt", "Teacher playbook content", {})

            # Playbook should come AFTER base prompt
            assert "Base prompt" in result
            assert "Teacher Playbook" in result
            assert result.index("Base prompt") < result.index("Teacher playbook content")


class TestFewShotExamples:
    """Test few-shot example extraction and budget enforcement."""

    def test_few_shot_budget_enforcement(self):
        """Verify that token budget is enforced for few-shot examples."""
        # Mock ExecutionContext with learning_usage containing examples
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_usage": {
                    "roles": {
                        "student": {
                            "entry_1": {
                                "cue_examples": ["Example 1" * 100, "Example 2" * 100],
                                "adoption_examples": [
                                    {"tool_name": "web_search", "status": "success"},
                                ],
                            },
                            "entry_2": {
                                "cue_examples": ["Example 3" * 100],
                                "adoption_examples": [],
                            },
                        }
                    }
                }
            }
            mock_ctx.return_value = mock_instance

            # Extract with small budget
            result = extract_few_shot_examples(
                metadata={},
                role="student",
                max_tokens=50,  # Very small budget
                redaction_patterns=[],
            )

            # Should limit content based on budget
            if result:
                # Rough estimate: 4 chars per token, so 50 tokens = 200 chars max
                assert len(result) < 300  # Allow some overhead for formatting

    def test_few_shot_no_examples(self):
        """Verify that None is returned when no examples available."""
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_usage": {
                    "roles": {
                        "student": {}
                    }
                }
            }
            mock_ctx.return_value = mock_instance

            result = extract_few_shot_examples(
                metadata={},
                role="student",
                max_tokens=500,
                redaction_patterns=[],
            )

            assert result is None

    def test_few_shot_example_formatting(self):
        """Verify that few-shot examples are properly formatted."""
        with patch.object(ExecutionContext, 'get') as mock_ctx:
            mock_instance = MagicMock()
            mock_instance.metadata = {
                "learning_usage": {
                    "roles": {
                        "student": {
                            "entry_test": {
                                "cue_examples": ["investigating latency issues"],
                                "adoption_examples": [
                                    {"tool_name": "metrics.query", "status": "success"},
                                ],
                            }
                        }
                    }
                }
            }
            mock_ctx.return_value = mock_instance

            result = extract_few_shot_examples(
                metadata={},
                role="student",
                max_tokens=500,
                redaction_patterns=[],
            )

            assert result is not None
            assert ">>> Few-Shot Examples >>>" in result
            assert ">>> End Few-Shot Examples >>>" in result
            assert "entry_test" in result
            assert "investigating latency issues" in result


class TestRedaction:
    """Test sensitive data redaction in examples."""

    def test_redaction_dates(self):
        """Verify that dates are redacted."""
        text = "Incident occurred on 2025-11-22 at 14:30"
        patterns = [r"\b\d{4}-\d{2}-\d{2}\b"]

        redacted = _redact_sensitive_data(text, patterns)

        assert "2025-11-22" not in redacted
        assert "[REDACTED]" in redacted

    def test_redaction_ids(self):
        """Verify that IDs are redacted."""
        text = "See INC-12345 and TICK-67890 for details"
        patterns = [r"\b[A-Z]{2,}-\d+\b"]

        redacted = _redact_sensitive_data(text, patterns)

        assert "INC-12345" not in redacted
        assert "TICK-67890" not in redacted
        assert "[REDACTED]" in redacted

    def test_redaction_multiple_patterns(self):
        """Verify that multiple patterns are applied."""
        text = "Incident INC-999 on 2025-01-15 affected CASE-123"
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b[A-Z]{2,}-\d+\b",
        ]

        redacted = _redact_sensitive_data(text, patterns)

        assert "2025-01-15" not in redacted
        assert "INC-999" not in redacted
        assert "CASE-123" not in redacted
        assert "[REDACTED]" in redacted

    def test_redaction_preserves_other_text(self):
        """Verify that non-sensitive text is preserved."""
        text = "Investigating latency issues with API endpoint"
        patterns = [r"\b\d{4}-\d{2}-\d{2}\b"]

        redacted = _redact_sensitive_data(text, patterns)

        assert redacted == text  # No changes expected

    def test_redaction_empty_patterns(self):
        """Verify that empty patterns return original text."""
        text = "INC-123 on 2025-01-01"
        patterns = []

        redacted = _redact_sensitive_data(text, patterns)

        assert redacted == text

    def test_redaction_invalid_pattern(self):
        """Verify that invalid patterns are skipped gracefully."""
        text = "Test content"
        patterns = [r"[invalid("]  # Invalid regex

        redacted = _redact_sensitive_data(text, patterns)

        # Should not crash, just skip invalid pattern
        assert redacted == text
