"""Worker process used by ``atlas env init`` to safely execute user code."""

from __future__ import annotations

import json
import typing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from importlib import import_module
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

from atlas.sdk.interfaces import AtlasAgentProtocol, AtlasEnvironmentProtocol, DiscoveryContext, TelemetryEmitterProtocol
from atlas.sdk.wrappers import StepwiseAgentAdapter


def _load_spec() -> dict[str, Any]:
    payload = sys.stdin.read()
    if not payload.strip():
        raise ValueError("Discovery worker received empty spec.")
    return json.loads(payload)


def _resolve_attr(module_path: str, qualname: str, *, project_root: Path | None = None) -> Any:
    if module_path.startswith("."):
        if project_root is None:
            raise ValueError("project_root is required for relative module imports")
        module = _load_relative_module(module_path, project_root)
    else:
        module = import_module(module_path)
    attr: Any = module
    for part in qualname.split("."):
        attr = getattr(attr, part)
    return attr


def _load_relative_module(module_path: str, project_root: Path) -> ModuleType:
    parts = [fragment for fragment in module_path.split(".") if fragment]
    if not parts:
        raise ValueError(f"Invalid relative module path: {module_path}")
    parts[0] = f".{parts[0]}"
    target_path = project_root.joinpath(*parts)
    if target_path.is_dir():
        file_path = target_path / "__init__.py"
    else:
        file_path = target_path.with_suffix(".py")
    if not file_path.exists():
        raise FileNotFoundError(f"Generated factory module not found at {file_path}")
    module_name = f"atlas_autogen_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    sys.modules[module_path] = module
    return module


def _schema_summary(value: Any) -> Dict[str, Any]:
    """Return a lightweight summary of a Python object for schema hints."""

    def _convert(obj: Any, depth: int = 0) -> Any:
        if depth > 2:
            return repr(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {
                str(key): _convert(val, depth + 1)
                for key, val in list(obj.items())[:5]
            }
        if isinstance(obj, (list, tuple, set)):
            seq = list(obj)
            return [_convert(item, depth + 1) for item in seq[:5]]
        return repr(obj)

    summary: Dict[str, Any] = {
        "python_type": type(value).__name__,
    }
    if isinstance(value, dict):
        summary["shape"] = {
            "kind": "mapping",
            "keys": list(value.keys())[:5],
        }
    elif isinstance(value, (list, tuple, set)):
        summary["shape"] = {
            "kind": "sequence",
            "length": len(value),
        }
        if value:
            summary["example_item"] = _convert(next(iter(value)))
    else:
        summary["example"] = _convert(value)
    return summary


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val, depth + 1) for key, val in list(value.items())[:8]}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in list(value)[:8]]
    return repr(value)


def _normalize_action(action: Any) -> tuple[Any, bool, dict[str, Any]]:
    submit = False
    metadata: dict[str, Any] = {}
    payload = action
    if isinstance(action, tuple) and action:
        payload = action[0]
        if len(action) > 1 and isinstance(action[1], bool):
            submit = action[1]
        if len(action) > 2 and isinstance(action[2], dict):
            metadata = dict(action[2])
    elif isinstance(action, dict):
        metadata = {k: v for k, v in action.items() if k not in {"action", "submit"}}
        if "action" in action:
            payload = action["action"]
        submit = bool(action.get("submit"))
    return payload, submit, metadata


class TelemetryCollector(TelemetryEmitterProtocol):
    """Collect telemetry emitted during discovery and runtime."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._agent_event_count = 0

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any] | None = None,
        *,
        origin: str | None = None,
    ) -> None:
        origin = origin or "agent"
        if origin == "agent":
            self._agent_event_count += 1
        event = {
            "type": event_type,
            "origin": origin,
            "payload": _json_safe(payload or {}),
            "timestamp": time.time(),
        }
        self._events.append(event)

    def emit_internal(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        self.emit(event_type, payload, origin="atlas")

    def flush(self) -> None:  # pragma: no cover - noop for interface compatibility
        return

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    @property
    def agent_emitted(self) -> bool:
        return self._agent_event_count > 0


@dataclass(slots=True)
class StepRecord:
    context: DiscoveryContext
    action: Any | None
    reward: float | None
    done: bool
    info: Dict[str, Any]
    submit: bool
    metadata: Dict[str, Any]


def _ensure_protocol(instance: Any, protocol: type) -> None:
    if not isinstance(instance, protocol):  # type: ignore[arg-type]
        raise TypeError(f"{instance!r} does not implement expected protocol {protocol.__name__}")


def _discovery_loop(
    env: AtlasEnvironmentProtocol,
    agent: AtlasAgentProtocol,
    task: str,
    *,
    emitter: TelemetryCollector,
) -> dict[str, Any]:
    history: list[DiscoveryContext] = []
    records: list[StepRecord] = []
    observation = env.reset(task=task)
    initial_context = DiscoveryContext(task=task, step_index=0, observation=observation)
    history.append(initial_context)
    emitter.emit_internal(
        "env_reset",
        {"observation": _json_safe(observation)},
    )
    plan_payload = agent.plan(task, observation, emit_event=emitter)
    emitter.emit_internal("plan_generated", {"plan": _json_safe(plan_payload)})
    done = False
    reward_total = 0.0
    step_index = 0
    latest_reward: float | None = None
    while not done:
        context = DiscoveryContext(
            task=task,
            step_index=step_index,
            observation=history[-1].observation,
            reward=latest_reward,
            done=False,
        )
        raw_action = agent.act(context, emit_event=emitter)
        action, submit, action_metadata = _normalize_action(raw_action)
        emitter.emit_internal(
            "env_action",
            {
                "action": _json_safe(action),
                "submit": submit,
                "metadata": _json_safe(action_metadata),
            },
        )
        step_method = typing.cast(Any, env.step)
        try:
            observation, reward, done, info = step_method(action, submit=submit)
        except TypeError:
            observation, reward, done, info = step_method(action)
        latest_reward = float(reward) if isinstance(reward, (int, float)) else None
        if latest_reward is not None:
            reward_total += latest_reward
        next_context = DiscoveryContext(
            task=task,
            step_index=step_index + 1,
            observation=observation,
            reward=latest_reward,
            done=done,
        )
        history.append(next_context)
        info_payload = info if isinstance(info, dict) else {"value": info}
        records.append(
            StepRecord(
                context=next_context,
                action=action,
                reward=latest_reward,
                done=done,
                info=info_payload,
                submit=submit,
                metadata=action_metadata,
            )
        )
        step_index += 1
        emitter.emit_internal(
            "env_step",
            {
                "step_index": step_index,
                "reward": latest_reward,
                "done": done,
                "info": _json_safe(info),
            },
        )
        if step_index > 256:
            raise RuntimeError("Discovery loop aborted: exceeded 256 steps without completion.")
    summary_context = history[-1]
    final_answer = agent.summarize(summary_context, history=history, emit_event=emitter)
    if isinstance(final_answer, str) and final_answer.strip():
        emitter.emit_internal("final_answer_submitted", {"text": final_answer.strip()})
    telemetry_payload = {
        "events": emitter.events,
        "agent_emitted": emitter.agent_emitted,
    }
    total_reward = reward_total if records else latest_reward
    reward_payload = {
        "total": total_reward,
        "last": latest_reward,
        "steps": sum(1 for record in records if record.reward is not None),
    }
    schema_payload = {
        "observation": _schema_summary(history[0].observation) if history else {},
        "action": _schema_summary(records[0].action) if records else {"python_type": "NoneType"},
        "reward": {"python_type": "float" if latest_reward is not None else "unknown"},
    }
    step_dump = [
        {
            "step_index": record.context.step_index,
            "observation": _json_safe(record.context.observation),
            "action": _json_safe(record.action),
            "reward": record.reward,
            "done": record.done,
            "info": _json_safe(record.info),
            "submit": record.submit,
            "metadata": _json_safe(record.metadata),
        }
        for record in records
    ]
    return {
        "plan": _json_safe(plan_payload),
        "final_answer": final_answer if isinstance(final_answer, str) else None,
        "telemetry": telemetry_payload,
        "reward": reward_payload,
        "schema": schema_payload,
        "history": step_dump,
    }


def _empty_result_payload() -> Dict[str, Any]:
    return {
        "plan": None,
        "final_answer": None,
        "telemetry": {"events": [], "agent_emitted": False},
        "reward": {"total": None, "last": None, "steps": 0},
        "schema": {},
        "history": [],
    }


def _import_and_build(role: str, module: str, qualname: str, *, kwargs: dict[str, Any] | None = None, project_root: Path | None = None) -> Any:
    kwargs = kwargs or {}
    attr = _resolve_attr(module, qualname, project_root=project_root)
    if isinstance(attr, type):
        return attr(**kwargs)
    if callable(attr):
        return attr(**kwargs)
    raise TypeError(f"Unsupported {role} target '{module}:{qualname}' – expected class or factory callable.")


def main() -> int:
    env_instance: AtlasEnvironmentProtocol | None = None
    agent_instance: AtlasAgentProtocol | None = None
    try:
        spec = _load_spec()
        project_root = Path(spec["project_root"]).resolve()
        sys.path.insert(0, str(project_root))
        src_dir = project_root / "src"
        if src_dir.exists():
            sys.path.insert(1, str(src_dir))
        extra_env = spec.get("env") or {}
        for key, value in extra_env.items():
            os.environ.setdefault(key, value)
        environment_spec = spec.get("environment") or {}
        agent_spec = spec.get("agent") or {}
        environment_factory_spec = spec.get("environment_factory")
        agent_factory_spec = spec.get("agent_factory")
        task = spec.get("task") or "Sample Atlas task."
        run_loop = bool(spec.get("run_discovery", True))
        skip_import = bool(spec.get("skip_import"))
        result_payload: dict[str, Any]
        if skip_import:
            env_instance = None
            agent_instance = None
            result_payload = _empty_result_payload()
            response = {"status": "ok", "result": result_payload}
            print(json.dumps(response))
            return 0
        env_kwargs = dict(environment_spec.get("init_kwargs") or {})
        agent_kwargs = dict(agent_spec.get("init_kwargs") or {})
        if environment_factory_spec:
            env_kwargs.update(environment_factory_spec.get("kwargs") or {})
        if agent_factory_spec:
            agent_kwargs.update(agent_factory_spec.get("kwargs") or {})

        if environment_factory_spec:
            env_module = environment_factory_spec.get("module")
            env_qualname = environment_factory_spec.get("qualname")
        else:
            env_module = environment_spec.get("module")
            env_qualname = environment_spec.get("qualname")
        if not env_module or not env_qualname:
            raise ValueError("Environment module/qualname not provided in discovery specification.")

        if agent_factory_spec:
            agent_module = agent_factory_spec.get("module")
            agent_qualname = agent_factory_spec.get("qualname")
        else:
            agent_module = agent_spec.get("module")
            agent_qualname = agent_spec.get("qualname")
        if not agent_module or not agent_qualname:
            raise ValueError("Agent module/qualname not provided in discovery specification.")

        env_instance = _import_and_build("environment", env_module, env_qualname, kwargs=env_kwargs, project_root=project_root)
        agent_instance = _import_and_build("agent", agent_module, agent_qualname, kwargs=agent_kwargs, project_root=project_root)

        if not callable(getattr(agent_instance, "plan", None)) or not callable(getattr(agent_instance, "summarize", None)):
            agent_instance = StepwiseAgentAdapter(agent_instance)

        _ensure_protocol(env_instance, AtlasEnvironmentProtocol)
        _ensure_protocol(agent_instance, AtlasAgentProtocol)
        emitter = TelemetryCollector()
        if run_loop:
            result_payload = _discovery_loop(env_instance, agent_instance, task, emitter=emitter)
        else:
            result_payload = _empty_result_payload()
        response = {
            "status": "ok",
            "result": result_payload,
        }
        print(json.dumps(response))
        return 0
    except Exception as exc:  # pragma: no cover - defensive failure path
        error_payload = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_payload))
        return 1
    finally:
        if env_instance is not None:
            try:
                close_method = getattr(env_instance, "close", None)
                if callable(close_method):
                    close_method()
            except Exception:
                pass
        if agent_instance is not None:
            try:
                close_method = getattr(agent_instance, "close", None)
                if callable(close_method):
                    close_method()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
