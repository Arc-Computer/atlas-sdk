"""Generic self-managed loop adapter bridging external agent environments."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Awaitable, Dict, Optional, Protocol, Sequence, runtime_checkable

from atlas.connectors.registry import (
    AdapterCapabilities,
    AdapterError,
    AdapterEventEmitter,
    AgentAdapter,
    register_adapter,
)
from atlas.config.models import AdapterType, PythonComponentConfig, SelfManagedAdapterConfig
from atlas.types import Plan, Step


@dataclass(slots=True)
class ManagedObservation:
    """Initial observation returned when resetting the environment."""

    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ManagedAction:
    """Action returned by the self-managed agent."""

    command: str
    submit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentStep:
    """Outcome from executing an action in the managed environment."""

    observation: Any
    reward: float | None
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ManagedEnvironment(Protocol):
    """Protocol implemented by managed environments."""

    async def areset(
        self,
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
    ) -> ManagedObservation:
        ...

    async def astep(
        self,
        action: ManagedAction,
        *,
        metadata: Dict[str, Any] | None = None,
    ) -> EnvironmentStep:
        ...

    async def aclose(self) -> None:
        ...


@runtime_checkable
class ManagedAgent(Protocol):
    """Protocol implemented by managed agents."""

    async def areset(
        self,
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        ...

    async def aplan(
        self,
        *,
        task: str,
        observation: ManagedObservation,
        metadata: Dict[str, Any] | None = None,
    ) -> Any:
        ...

    async def aact(
        self,
        observation: Any,
        *,
        step_index: int,
        metadata: Dict[str, Any] | None = None,
    ) -> ManagedAction:
        ...

    async def asummarize(
        self,
        trajectory: Sequence[Dict[str, Any]],
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Optional[str]:
        ...


def _maybe_await(value: Any) -> Awaitable[Any]:
    if inspect.isawaitable(value):
        return value  # type: ignore[return-value]
    async def _wrapper() -> Any:
        return value
    return _wrapper()


def _load_component(component: PythonComponentConfig) -> Any:
    module = import_module(component.import_path)
    target = getattr(module, component.attribute) if component.attribute else module
    if inspect.isclass(target):
        return target(**component.options)
    if callable(target):
        return target(**component.options)
    if component.options:
        raise AdapterError(f"component '{component.import_path}' is not callable but received options")
    return target


class SelfManagedLoopAdapter(AgentAdapter):
    """Adapter coordinating a self-managed agent/environment pair."""

    def __init__(self, config: SelfManagedAdapterConfig):
        self._config = config
        self._telemetry_enabled = bool(config.telemetry_stream)
        self._max_iterations = config.max_iterations
        self._default_description = (
            config.plan_description
            or "Execute the partner-managed control loop exactly once and return the final answer."
        )
        self._environment: ManagedEnvironment | None = None
        self._agent: ManagedAgent | None = None
        self._emit_event: AdapterEventEmitter | None = None
        self._session_state: Dict[str, Any] = {}

    async def aopen_session(
        self,
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
        emit_event: AdapterEventEmitter | None = None,
    ) -> AdapterCapabilities:
        environment = await _maybe_await(_load_component(self._config.environment))
        agent = await _maybe_await(_load_component(self._config.agent))
        if not isinstance(environment, ManagedEnvironment):
            raise AdapterError("environment does not implement the ManagedEnvironment protocol")
        if not isinstance(agent, ManagedAgent):
            raise AdapterError("agent does not implement the ManagedAgent protocol")
        self._environment = environment
        self._agent = agent
        self._emit_event = emit_event if self._telemetry_enabled else None
        self._session_state = {"metadata": metadata or {}}
        if hasattr(agent, "set_event_emitter") and callable(getattr(agent, "set_event_emitter")):
            await _maybe_await(agent.set_event_emitter(self._emit_event))  # type: ignore[attr-defined]
        if hasattr(environment, "set_event_emitter") and callable(getattr(environment, "set_event_emitter")):
            await _maybe_await(environment.set_event_emitter(self._emit_event))  # type: ignore[attr-defined]
        return AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=self._telemetry_enabled)

    async def aplan(self, task: str, metadata: Dict[str, Any] | None = None) -> Plan:
        environment = self._require_environment()
        agent = self._require_agent()
        session_meta = self._session_state.get("metadata", {}).copy()
        session_meta.update(metadata or {})
        await _maybe_await(agent.areset(task=task, metadata=session_meta))
        observation = await _maybe_await(environment.areset(task=task, metadata=session_meta))
        self._session_state["initial_observation"] = observation
        self._session_state["plan_metadata"] = {"task": task, "initial_metadata": observation.metadata}
        plan_payload = await _maybe_await(agent.aplan(task=task, observation=observation, metadata=session_meta))
        if plan_payload:
            try:
                plan = Plan.model_validate(plan_payload)
                plan = plan.model_copy(update={"execution_mode": "single_shot"})
            except Exception as exc:
                raise AdapterError(f"agent returned invalid plan payload: {exc}") from exc
        else:
            default_step = Step(
                id=1,
                description=self._default_description,
                tool=None,
                tool_params=None,
                depends_on=[],
            )
            plan = Plan(steps=[default_step], execution_mode="single_shot")
        self._session_state["plan"] = plan
        return plan

    async def aexecute(
        self,
        task: str,
        plan: Dict[str, Any],
        step: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> StudentStepResult:
        environment = self._require_environment()
        agent = self._require_agent()
        plan_model = Plan.model_validate(plan) if not isinstance(plan, Plan) else plan
        step_model = Step.model_validate(step) if not isinstance(step, Step) else step
        session_meta = self._session_state.get("metadata", {}).copy()
        session_meta.update(metadata or {})
        observation: ManagedObservation = self._session_state.get("initial_observation")
        if observation is None:
            observation = await _maybe_await(environment.areset(task=task, metadata=session_meta))
            self._session_state["initial_observation"] = observation
        trajectory: list[Dict[str, Any]] = []
        submit_action: ManagedAction | None = None
        iteration = 0
        env_step: EnvironmentStep | None = None
        while True:
            if self._max_iterations is not None and iteration >= self._max_iterations:
                raise AdapterError(f"self-managed loop exceeded max_iterations={self._max_iterations}")
            action = await _maybe_await(
                agent.aact(
                    observation.content,
                    step_index=iteration,
                    metadata={"step": iteration, **observation.metadata},
                )
            )
            if not isinstance(action, ManagedAction):
                raise AdapterError("agent.aact must return ManagedAction instances")
            if action.command is None or not str(action.command).strip():
                raise AdapterError("agent returned empty command")
            env_step = await _maybe_await(
                environment.astep(
                    action,
                    metadata={"step": iteration, **(action.metadata or {})},
                )
            )
            if not isinstance(env_step, EnvironmentStep):
                raise AdapterError("environment.astep must return EnvironmentStep instances")
            trajectory_entry = {
                "step": iteration,
                "action": action.command,
                "submit": action.submit,
                "observation": env_step.observation,
                "reward": env_step.reward,
                "done": env_step.done,
                "info": env_step.info,
            }
            trajectory.append(trajectory_entry)
            if self._emit_event is not None:
                await self._emit_event(
                    {
                        "event": "env_action",
                        "payload": {
                            "command": action.command,
                            "observation": env_step.observation,
                            "info": env_step.info,
                        },
                        "reason": action.metadata.get("reason") if action.metadata else None,
                        "step": iteration,
                        "metadata": {
                            "submit": action.submit,
                            "reward": env_step.reward,
                        },
                    }
                )
            if action.submit or env_step.done:
                submit_action = action
                break
            observation = ManagedObservation(
                content=env_step.observation,
                metadata=env_step.info,
            )
            iteration += 1
        if env_step is None:
            raise AdapterError("self-managed loop produced no environment steps")
        summary = await _maybe_await(
            agent.asummarize(
                trajectory=trajectory,
                task=task,
                metadata=session_meta,
            )
        )
        final_output = summary if isinstance(summary, str) and summary.strip() else str(env_step.observation)
        trace = {
            "plan": plan_model.model_dump(),
            "step": step_model.model_dump(),
            "trajectory": trajectory,
            "submit": submit_action.command if submit_action else None,
        }
        metadata_payload = {
            "trajectory": trajectory,
            "environment_info": self._session_state.get("plan_metadata", {}).get("initial_metadata"),
            "submit_action": submit_action.command if submit_action else None,
            "reward": env_step.reward,
            "iterations": iteration + 1,
            "final_info": env_step.info,
        }
        return {
            "trace": json.dumps(trace, ensure_ascii=False),
            "output": final_output,
            "messages": [],
            "metadata": metadata_payload,
            "status": "completed" if env_step.done else "unknown",
            "artifacts": {"trajectory": trajectory},
            "deliverable": final_output,
        }

    async def asynthesize(
        self,
        task: str,
        plan: Dict[str, Any],
        step_results: Sequence[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        if not step_results:
            return ""
        last_result = step_results[-1]
        deliverable = last_result.get("deliverable")
        if isinstance(deliverable, str) and deliverable.strip():
            return deliverable
        return str(last_result.get("output", ""))

    async def aclose(self) -> None:
        environment = self._environment
        self._environment = None
        self._agent = None
        if environment and self._config.auto_close_environment:
            await _maybe_await(environment.aclose())
        self._session_state.clear()

    def _require_environment(self) -> ManagedEnvironment:
        if self._environment is None:
            raise AdapterError("session has no active environment; ensure aopen_session was called")
        return self._environment

    def _require_agent(self) -> ManagedAgent:
        if self._agent is None:
            raise AdapterError("session has no active agent; ensure aopen_session was called")
        return self._agent


register_adapter(AdapterType.SELF_MANAGED, SelfManagedLoopAdapter)

__all__ = [
    "ManagedAction",
    "ManagedAgent",
    "ManagedEnvironment",
    "ManagedObservation",
    "EnvironmentStep",
    "SelfManagedLoopAdapter",
]
