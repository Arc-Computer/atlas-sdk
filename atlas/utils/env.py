"""Helpers for loading environment variables from local configuration files."""

from __future__ import annotations

import os
import warnings
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


def is_offline_mode() -> bool:
    """
    Check if Atlas is running in offline mode.
    
    Supports both ATLAS_OFFLINE_MODE (new) and ATLAS_FAKE_LLM (deprecated).
    Returns True if either variable is set to a truthy value.
    """
    offline_mode = os.getenv("ATLAS_OFFLINE_MODE", "0") not in {"0", "", "false", "False"}
    if not offline_mode:
        fake_llm = os.getenv("ATLAS_FAKE_LLM", "0") not in {"0", "", "false", "False"}
        if fake_llm:
            warnings.warn(
                "ATLAS_FAKE_LLM is deprecated. Use ATLAS_OFFLINE_MODE=1 instead.",
                DeprecationWarning,
                stacklevel=2
            )
            offline_mode = True
    return offline_mode


__all__ = ["load_dotenv_if_available", "is_offline_mode"]
