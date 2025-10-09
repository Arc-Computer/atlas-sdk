"""Opinionated system prompts and prompt dataclasses."""

from .system import (
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
    build_student_prompts,
    build_teacher_prompts,
)

__all__ = [
    "RewrittenStudentPrompts",
    "RewrittenTeacherPrompts",
    "build_student_prompts",
    "build_teacher_prompts",
]
