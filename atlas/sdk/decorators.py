"""Decorators used to explicitly mark Atlas environments and agents."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

T = TypeVar("T", bound=type)


def _mark_role(cls: T, role: str, name: str | None = None) -> T:
    setattr(cls, "__atlas_role__", role)
    if name:
        setattr(cls, "__atlas_label__", name)
    return cls


def environment(cls: Optional[T] = None, *, name: Optional[str] = None) -> Callable[[T], T] | T:
    """Decorator tagging a class as an Atlas-compatible environment."""

    def decorator(target: T) -> T:
        return _mark_role(target, "environment", name)

    if cls is None:
        return decorator
    return decorator(cls)


def agent(cls: Optional[T] = None, *, name: Optional[str] = None) -> Callable[[T], T] | T:
    """Decorator tagging a class as an Atlas-compatible self-managed agent."""

    def decorator(target: T) -> T:
        return _mark_role(target, "agent", name)

    if cls is None:
        return decorator
    return decorator(cls)
