"""JSON Schema builder for playbook entry learning synthesis."""

from __future__ import annotations

from typing import Any, Dict


def build_playbook_entry_schema() -> Dict[str, Any]:
    """Build JSON Schema for playbook_entry.v1 structure.
    
    This schema matches the structure defined in atlas/learning/prompts.py
    and is used for Gemini structured outputs to enforce type safety and
    schema validation at the API level.
    
    Returns:
        JSON Schema dictionary compatible with Gemini's response_json_schema format
    """
    return {
        "type": "object",
        "properties": {
            "version": {
                "type": "string",
                "const": "playbook_entry.v1",
                "description": "Schema version identifier"
            },
            "student_pamphlet": {
                "type": ["string", "null"],
                "description": "Updated student learning pamphlet text or null if unchanged"
            },
            "teacher_pamphlet": {
                "type": ["string", "null"],
                "description": "Updated teacher learning pamphlet text or null if unchanged"
            },
            "playbook_entries": {
                "type": "array",
                "description": "List of playbook entries to add or update",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": ["string", "null"],
                            "description": "Unique identifier for the entry, or null for new entries"
                        },
                        "audience": {
                            "type": "string",
                            "enum": ["student", "teacher"],
                            "description": "Target audience for this playbook entry"
                        },
                        "cue": {
                            "type": "object",
                            "description": "Machine-detectable trigger pattern",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["regex", "keyword", "predicate"],
                                    "description": "Type of cue pattern"
                                },
                                "pattern": {
                                    "type": "string",
                                    "description": "Machine-detectable trigger pattern (max 150 chars)"
                                },
                                "description": {
                                    "type": ["string", "null"],
                                    "description": "Optional human-readable explanation"
                                }
                            },
                            "required": ["type", "pattern"]
                        },
                        "action": {
                            "type": "object",
                            "description": "Action to take when cue is detected",
                            "properties": {
                                "imperative": {
                                    "type": "string",
                                    "description": "Imperative verb phrasing describing the action (max 120 chars)"
                                },
                                "runtime_handle": {
                                    "type": ["string", "null"],
                                    "description": "Runtime handle/tool name from available_runtime_handles, or null if no tools"
                                },
                                "tool_name": {
                                    "type": ["string", "null"],
                                    "description": "Optional tool name"
                                },
                                "arguments": {
                                    "type": ["object", "null"],
                                    "description": "Optional tool arguments"
                                }
                            },
                            "required": ["imperative"]
                        },
                        "expected_effect": {
                            "type": "string",
                            "description": "Explanation of why this action improves outcomes (max 200 chars)"
                        },
                        "scope": {
                            "type": "object",
                            "description": "Scope and constraints for when this entry applies",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["reinforcement", "differentiation"],
                                    "description": "Whether this reinforces existing behavior or introduces new strategy"
                                },
                                "constraints": {
                                    "type": "string",
                                    "description": "Boundaries and applicability constraints (max 250 chars)"
                                },
                                "applies_when": {
                                    "type": ["string", "null"],
                                    "description": "Optional condition for when this entry applies"
                                }
                            },
                            "required": ["category", "constraints"]
                        },
                        "metadata": {
                            "type": ["object", "null"],
                            "description": "Optional free-form metadata"
                        }
                    },
                    "required": ["audience", "cue", "action", "expected_effect", "scope"]
                }
            },
            "session_student_learning": {
                "type": ["string", "null"],
                "description": "Brief takeaway from this session for student (optional)"
            },
            "session_teacher_learning": {
                "type": ["string", "null"],
                "description": "Teacher intervention note from this session (optional)"
            },
            "metadata": {
                "type": ["object", "null"],
                "description": "Optional metadata including synthesis reasoning and validation notes"
            }
        },
        "required": ["version"]
    }


__all__ = ["build_playbook_entry_schema"]

