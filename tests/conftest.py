"""Pytest configuration to stub out optional heavy dependencies."""

from __future__ import annotations

import importlib.abc
import os
import sys


os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that prevents importing torch in test environments."""

    def find_spec(self, fullname: str, path, target=None):  # noqa: D401, ANN001, ANN002, ANN003
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch is disabled in this test environment")
        return None


sys.meta_path.insert(0, _BlockTorchFinder())
