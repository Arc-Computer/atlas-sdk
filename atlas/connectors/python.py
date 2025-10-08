"""Python adapter allowing local callables to serve as BYOA agents."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
from typing import Any
from typing import Dict

from atlas.connectors.registry import AdapterError
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import register_adapter
from atlas.config.models import AdapterType
from atlas.config.models import PythonAdapterConfig

class PythonAdapter(AgentAdapter):
    """Adapter that calls a user supplied Python function."""

    def __init__(self, config: PythonAdapterConfig):
        self._config = config
        self._callable = self._load_callable()
    def _load_callable(self):
        module_path = self._config.import_path
        working_dir = self._config.working_directory
        if working_dir and working_dir not in sys.path:
            sys.path.insert(0, working_dir)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise AdapterError(f"unable to import module '{module_path}'") from exc
        attr_name = self._config.attribute or "main"
        try:
            target = getattr(module, attr_name)
        except AttributeError as exc:
            raise AdapterError(f"attribute '{attr_name}' not found in module '{module_path}'") from exc
        if not callable(target):
            raise AdapterError(f"attribute '{attr_name}' is not callable")
        return target
    async def _normalise_result(self, result: Any) -> str:
        if inspect.isasyncgen(result):
            if not self._config.allow_generator:
                raise AdapterError("generator outputs are disabled for this adapter")
            parts = []
            async for item in result:
                parts.append(str(item))
            return "".join(parts)
        if inspect.isgenerator(result):
            if not self._config.allow_generator:
                raise AdapterError("generator outputs are disabled for this adapter")
            return "".join(str(item) for item in result)
        if isinstance(result, bytes):
            return result.decode("utf-8")
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)
    def _call_sync(self, prompt: str, metadata: Dict[str, Any] | None) -> Any:
        try:
            return self._callable(prompt=prompt, metadata=metadata)
        except Exception as exc:
            raise AdapterError(f"python adapter callable raised an exception: {exc}") from exc
    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        call_metadata = metadata or {}
        if self._config.llm:
            call_metadata["llm_config"] = self._config.llm.model_dump()
        func = self._callable
        if inspect.iscoroutinefunction(func):
            try:
                result = await func(prompt=prompt, metadata=call_metadata)
            except Exception as exc:
                raise AdapterError(f"python adapter coroutine raised an exception: {exc}") from exc
        else:
            result = await asyncio.to_thread(self._call_sync, prompt, call_metadata)
        return await self._normalise_result(result)

register_adapter(AdapterType.PYTHON, PythonAdapter)

__all__ = ["PythonAdapter"]
