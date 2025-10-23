"""Helpers for loading environment variables from local configuration files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_dotenv_if_available(dotenv_path: Path | str | None = None) -> bool:
    """Load variables from a `.env` file if python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - dependency declared but defensive
        return False
    kwargs: dict[str, Any] = {}
    if dotenv_path:
        kwargs["dotenv_path"] = str(dotenv_path)
    load_dotenv(**kwargs)
    return True


__all__ = ["load_dotenv_if_available"]
