"""Helpers for loading environment variables from local configuration files."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def load_dotenv_if_available() -> bool:
    """Load variables from a `.env` file if python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - dependency declared but defensive
        return False
    load_dotenv()
    return True


__all__ = ["load_dotenv_if_available"]
