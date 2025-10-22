"""Worker process used by ``atlas env init`` to safely execute user code."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple

from atlas.sdk.interfaces import AtlasAgentProtocol, AtlasEnvironmentProtocol, DiscoveryContext, TelemetryEmitterProtocol


def _load_spec() -> dict[str, Any]:
    payload = sys.stdin.read()
    if not payload.strip():
        raise ValueError("Discovery worker received empty spec.")
    return json.loads(payload)


def _resolve_attr(module_path: str, qualname: str) -> Any:
    module = import_module(module_path)
    attr: Any = module
    for part in qualname.split("."):
        attr = getattr(attr, part)
    return attr


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

    summary = {
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
        action = agent.act(context, emit_event=emitter)
        emitter.emit_internal("env_action", {"action": _json_safe(action)})
        observation, reward, done, info = env.step(action)
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
        records.append(
            StepRecord(
                context=next_context,
                action=action,
                reward=latest_reward,
                done=done,
                info=_json_safe(info),
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
    reward_payload = {
        "total": reward_total if reward_total else latest_reward,
        "last": latest_reward,
        "steps": sum(1 for record in records if record.reward is not None),
    }
    schema_payload = {
        "observation": _schema_summary(history[0].observation),
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
            "info": record.info,
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


def _import_and_build(role: str, module: str, qualname: str) -> Any:
    attr = _resolve_attr(module, qualname)
    if isinstance(attr, type):
        return attr()
    if callable(attr):
        return attr()
    raise TypeError(f"Unsupported {role} target '{module}:{qualname}' – expected class or factory callable.")


def main() -> int:
    env_instance: AtlasEnvironmentProtocol | None = None
    agent_instance: AtlasAgentProtocol | None = None
    try:
        spec = _load_spec()
        project_root = Path(spec["project_root"]).resolve()
        sys.path.insert(0, str(project_root))
        extra_env = spec.get("env") or {}
        for key, value in extra_env.items():
            os.environ.setdefault(key, value)
        environment_spec = spec["environment"]
        agent_spec = spec["agent"]
        task = spec.get("task") or "Sample Atlas task."
        run_loop = bool(spec.get("run_discovery", True))
        env_instance = _import_and_build("environment", environment_spec["module"], environment_spec["qualname"])
        agent_instance = _import_and_build("agent", agent_spec["module"], agent_spec["qualname"])
        _ensure_protocol(env_instance, AtlasEnvironmentProtocol)
        _ensure_protocol(agent_instance, AtlasAgentProtocol)
        emitter = TelemetryCollector()
        result_payload: dict[str, Any]
        if run_loop:
            result_payload = _discovery_loop(env_instance, agent_instance, task, emitter=emitter)
        else:
            result_payload = {
                "plan": None,
                "final_answer": None,
                "telemetry": {"events": [], "agent_emitted": False},
                "reward": {"total": None, "last": None, "steps": 0},
                "schema": {},
                "history": [],
            }
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
