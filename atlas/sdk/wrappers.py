"""Adapters that retrofit third-party agents for Atlas discovery/runtime."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from atlas.sdk.interfaces import AtlasAgentProtocol, DiscoveryContext, TelemetryEmitterProtocol


def _interpret_action_payload(payload: Any) -> tuple[Any, bool, dict[str, Any]]:
    submit = False
    metadata: dict[str, Any] = {}
    action = payload
    if isinstance(payload, tuple) and payload:
        action = payload[0]
        if len(payload) > 1 and isinstance(payload[1], bool):
            submit = payload[1]
        if len(payload) > 2 and isinstance(payload[2], dict):
            metadata = dict(payload[2])
    elif isinstance(payload, dict):
        metadata = {k: v for k, v in payload.items() if k not in {"action", "submit"}}
        action = payload.get("action", action)
        submit = bool(payload.get("submit"))
    return action, submit, metadata


class StepwiseAgentAdapter(AtlasAgentProtocol):
    """Wraps an agent that only implements ``act`` (and optionally ``reset``/``plan``/``summarize``)."""

    def __init__(self, agent: object) -> None:
        self._agent = agent
        self._final_answer: Optional[str] = None

    # Public AtlasAgentProtocol -------------------------------------------------

    def plan(
        self,
        task: str,
        observation: Any,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Any:
        plan_result = self._call_optional("plan", task, observation, emit_event=emit_event)
        if plan_result is not None:
            return plan_result
        reset_result = self._call_optional("reset")
        if reset_result is None:
            self._call_optional("reset", task)
        return {
            "execution": "stepwise",
            "note": "Auto-generated plan by Atlas stepwise adapter.",
        }

    def act(
        self,
        context: DiscoveryContext,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Any:
        observation = context.observation
        result = self._call_optional("act", context, emit_event=emit_event)
        if result is None:
            result = self._call_optional("act", observation, emit_event=emit_event)
        if result is None:
            raise TypeError("Wrapped agent does not expose an act(...) method with compatible signature.")
        action, submit, metadata = _interpret_action_payload(result)
        if submit and isinstance(action, str):
            self._final_answer = action
        payload = {"action": action, "submit": submit}
        if metadata:
            payload["metadata"] = metadata
        return payload

    def summarize(
        self,
        context: DiscoveryContext,
        *,
        history: Iterable[DiscoveryContext] | None = None,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Optional[str]:
        summary = self._call_optional("summarize", context, history=history, emit_event=emit_event)
        if summary is not None:
            if isinstance(summary, str):
                self._final_answer = summary
            return summary
        return self._final_answer or ""

    # Internal helpers ---------------------------------------------------------

    def _call_optional(self, name: str, *args: Any, **kwargs: Any) -> Any | None:
        target = getattr(self._agent, name, None)
        if not callable(target):
            return None
        try:
            return target(*args, **kwargs)
        except TypeError:
            # Retry without keyword arguments if possible.
            if kwargs:
                try:
                    return target(*args)
                except TypeError:
                    return None
            return None

    # Convenience passthroughs -------------------------------------------------

    def __getattr__(self, item: str) -> Any:
        return getattr(self._agent, item)
