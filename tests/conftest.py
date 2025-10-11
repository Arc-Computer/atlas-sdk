"""Pytest configuration to stub out optional heavy dependencies."""

from __future__ import annotations

import asyncio
import importlib.abc
import os
import sys

import pytest


os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that prevents importing torch in test environments."""

    def find_spec(self, fullname: str, path, target=None):  # noqa: D401, ANN001, ANN002, ANN003
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch is disabled in this test environment")
        return None


sys.meta_path.insert(0, _BlockTorchFinder())

DEFAULT_POSTGRES_DSN = "postgresql://atlas:atlas@localhost:5433/atlas_arc_demo"
_POSTGRES_AVAILABLE: bool | None = None


async def _attempt_postgres(dsn: str) -> bool:
    try:
        import asyncpg
    except ModuleNotFoundError:
        return False
    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=1.5)
    except Exception:
        return False
    else:
        await conn.close()
        return True


def _postgres_available() -> bool:
    global _POSTGRES_AVAILABLE
    if _POSTGRES_AVAILABLE is not None:
        return _POSTGRES_AVAILABLE
    dsn = os.getenv("STORAGE__DATABASE_URL", DEFAULT_POSTGRES_DSN)
    try:
        _POSTGRES_AVAILABLE = asyncio.run(_attempt_postgres(dsn))
    except RuntimeError:
        # Fallback for event loop already running; assume unavailable to avoid flaky behaviour.
        _POSTGRES_AVAILABLE = False
    return _POSTGRES_AVAILABLE


def pytest_configure(config):
    config.addinivalue_line("markers", "postgres: marks tests that require a local PostgreSQL instance")


def pytest_runtest_setup(item):
    if "postgres" in item.keywords and not _postgres_available():
        pytest.skip("PostgreSQL DSN not reachable; skipping postgres-marked test")

