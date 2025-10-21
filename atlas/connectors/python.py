"""Python adapter allowing local callables to serve as BYOA agents."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, MutableMapping
from uuid import uuid4

from atlas.connectors.registry import AdapterError
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import AgentSession
from atlas.connectors.registry import SessionContext
from atlas.connectors.registry import StatelessSession
from atlas.connectors.registry import register_adapter
from atlas.config.models import AdapterType
from atlas.config.models import PythonAdapterConfig


@dataclass(slots=True)
class _StatefulSpec:
    factory: Callable[[], Any]
    open_hook: str | None
    step_hook: str
    close_hook: str | None


class SessionResourcePool:
    """Simple async pool for sharing resources across stateful sessions."""

    def __init__(self, factory: Callable[[], Awaitable[Any] | Any], *, max_size: int = 8) -> None:
        self._factory = factory
        self._max_size = max(1, max_size)
        self._queue: asyncio.Queue[Any] = asyncio.Queue(max_size)
        self._lock = asyncio.Lock()
        self._created = 0

    async def acquire(self) -> Any:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        async with self._lock:
            if self._created < self._max_size:
                self._created += 1
                return await _maybe_await(self._factory())
        return await self._queue.get()

    async def release(self, resource: Any) -> None:
        if self._queue.full():
            return
        await self._queue.put(resource)

    async def drain(self, disposer: Callable[[Any], Awaitable[None] | None] | None = None) -> None:
        while not self._queue.empty():
            resource = await self._queue.get()
            if disposer is not None:
                await _maybe_await(disposer(resource))


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


class _PythonStatefulSession(AgentSession):
    """Session wrapper for stateful Python classes implementing hook methods."""

    def __init__(self, spec: _StatefulSpec, context: SessionContext) -> None:
        self._spec = spec
        self._context = context
        self._instance: Any | None = None
        self._metadata: MutableMapping[str, Any] = {}
        self._session_id = context.session_id or f"python-{uuid4()}"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def open(self) -> None:
        self._instance = self._spec.factory()
        if inspect.isawaitable(self._instance):
            self._instance = await self._instance
        if self._spec.open_hook and self._instance is not None:
            hook = getattr(self._instance, self._spec.open_hook)
            result = await _maybe_await(hook(self._context))
            if isinstance(result, Mapping):
                self._metadata.update(result)
            elif result is not None:
                self._metadata["open_return"] = result

    async def step(self, payload: str, metadata: Dict[str, Any] | None = None) -> Any:
        if self._instance is None:
            await self.open()
        assert self._instance is not None  # for type checkers
        hook = getattr(self._instance, self._spec.step_hook)
        call_kwargs = {}
        signature = inspect.signature(hook)
        if "prompt" in signature.parameters:
            call_kwargs["prompt"] = payload
        if "payload" in signature.parameters and "prompt" not in signature.parameters:
            call_kwargs["payload"] = payload
        if "metadata" in signature.parameters:
            call_kwargs["metadata"] = metadata
        if not call_kwargs:
            call_kwargs["prompt"] = payload
        result = await _maybe_await(hook(**call_kwargs))
        return result

    async def close(self, reason: str | None = None) -> Mapping[str, Any] | None:
        if self._instance is None:
            return self._metadata
        if not self._spec.close_hook:
            return self._metadata
        hook = getattr(self._instance, self._spec.close_hook)
        signature = inspect.signature(hook)
        call_kwargs = {}
        if "reason" in signature.parameters:
            call_kwargs["reason"] = reason
        if "context" in signature.parameters:
            call_kwargs["context"] = self._context
        result = await _maybe_await(hook(**call_kwargs))
        if isinstance(result, Mapping):
            self._metadata.update(result)
        return self._metadata


class PythonAdapter(AgentAdapter):
    """Adapter that calls a user supplied Python function or session-aware class."""

    def __init__(self, config: PythonAdapterConfig):
        self._config = config
        self._target = self._load_callable()
        self._stateful_spec = self._detect_stateful_target(self._target)
        if self._stateful_spec is not None:
            self.supports_sessions = True

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

    def _detect_stateful_target(self, target: Any) -> _StatefulSpec | None:
        if not inspect.isclass(target):
            return None
        open_hook = self._first_hook(target, ["on_open", "open", "start"])
        step_hook = self._first_hook(target, ["on_step", "step", "run", "__call__"])
        close_hook = self._first_hook(target, ["on_close", "close", "shutdown", "stop"])
        if step_hook is None:
            return None
        return _StatefulSpec(factory=target, open_hook=open_hook, step_hook=step_hook, close_hook=close_hook)

    def _first_hook(self, target: Any, names: list[str]) -> str | None:
        for name in names:
            if hasattr(target, name) and callable(getattr(target, name)):
                return name
        return None

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
            return self._target(prompt=prompt, metadata=metadata)
        except Exception as exc:  # pragma: no cover - best effort
            raise AdapterError(f"python adapter callable raised an exception: {exc}") from exc

    async def _call_stateful_once(self, prompt: str, metadata: Dict[str, Any] | None) -> Any:
        context = self._context_from_metadata(metadata)
        session = await self.open_session(context)
        try:
            return await session.step(prompt, metadata)
        finally:
            await session.close(reason="adhoc")

    def _context_from_metadata(self, metadata: Dict[str, Any] | None) -> SessionContext:
        payload = dict(metadata or {})
        task_id = str(payload.get("task_id") or payload.get("task") or uuid4())
        execution_mode = str(payload.get("mode") or payload.get("execution_mode") or "adhoc")
        tool_meta = payload.get("tool_metadata")
        user_ctx = payload.get("user_context")
        return SessionContext(task_id=task_id, execution_mode=execution_mode, tool_metadata=tool_meta, user_context=user_ctx)

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        call_metadata = dict(metadata or {})
        if self._config.llm:
            call_metadata.setdefault("llm_config", self._config.llm.model_dump())
        if self._stateful_spec is not None:
            result = await self._call_stateful_once(prompt, call_metadata)
            return await self._normalise_result(result)
        func = self._target
        if inspect.iscoroutinefunction(func):
            try:
                result = await func(prompt=prompt, metadata=call_metadata)
            except Exception as exc:
                raise AdapterError(f"python adapter coroutine raised an exception: {exc}") from exc
        else:
            result = await asyncio.to_thread(self._call_sync, prompt, call_metadata)
        return await self._normalise_result(result)

    async def open_session(self, context: SessionContext) -> AgentSession:
        if self._stateful_spec is None:
            return StatelessSession(self, context)
        session = _PythonStatefulSession(self._stateful_spec, context)
        await session.open()
        return session


register_adapter(AdapterType.PYTHON, PythonAdapter)

__all__ = ["PythonAdapter", "SessionResourcePool"]
