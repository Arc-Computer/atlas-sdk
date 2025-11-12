"""Lightweight wrapper around litellm for synchronous and asynchronous calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import litellm  # type: ignore[import-untyped]
    _LITELLM_ERROR = None
    litellm.drop_params = True
except ModuleNotFoundError as exc:
    litellm = None  # type: ignore[assignment]
    _LITELLM_ERROR = exc

from atlas.config.models import LLMParameters
from atlas.utils.env import is_offline_mode


@dataclass
class LLMResponse:
    content: str
    raw: Any
    reasoning: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, parameters: LLMParameters) -> None:
        self._params = parameters
        self._mock_mode = is_offline_mode()

    @property
    def model(self) -> str:
        return self._params.model

    async def acomplete(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._mock_mode:
            return self._mock_response(messages, response_format)
        self._ensure_client()
        kwargs = self._prepare_kwargs(messages, response_format, overrides)
        result = await litellm.acompletion(**kwargs)
        content, reasoning = self._extract_content(result)
        return LLMResponse(content=content, reasoning=reasoning, raw=result)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._mock_mode:
            return self._mock_response(messages, response_format)
        self._ensure_client()
        kwargs = self._prepare_kwargs(messages, response_format, overrides)
        result = litellm.completion(**kwargs)
        content, reasoning = self._extract_content(result)
        return LLMResponse(content=content, reasoning=reasoning, raw=result)

    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model.
        
        Args:
            model: Model identifier string
            
        Returns:
            True if model is a Gemini model, False otherwise
        """
        return model.startswith("gemini/") or model.startswith("google/")

    def _prepare_kwargs(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        import os
        
        params = self._params
        api_key = os.getenv(params.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable '{params.api_key_env}' is not set")
        overrides = overrides or {}
        kwargs: dict[str, Any] = {"model": params.model, "messages": list(messages), "api_key": api_key}
        if params.api_base:
            kwargs["api_base"] = params.api_base
        if params.organization:
            kwargs["organization"] = params.organization
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        kwargs["timeout"] = params.timeout_seconds

        extra_headers = dict(params.additional_headers)
        override_headers = overrides.pop("extra_headers", None)
        if override_headers:
            extra_headers.update(override_headers)

        extra_body = dict(overrides.pop("extra_body", {}) or {})
        supports_reasoning = False
        if litellm is not None and hasattr(litellm, "supports_reasoning"):
            try:
                supports_reasoning = bool(litellm.supports_reasoning(params.model))
            except Exception:
                supports_reasoning = False
        if supports_reasoning and params.reasoning_effort:
            extra_body.setdefault("reasoning_effort", params.reasoning_effort)

        # Handle Gemini structured outputs
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            if self._is_gemini_model(params.model):
                # Use Gemini structured outputs via extra_body
                # response_json_schema should be provided via overrides["extra_body"]
                extra_body.setdefault("response_mime_type", "application/json")
                # Don't set response_format for Gemini - it's not supported
            else:
                # Use OpenAI-style response_format for non-Gemini models
                kwargs["response_format"] = response_format

        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if extra_body:
            kwargs["extra_body"] = extra_body

        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    def _ensure_client(self) -> None:
        if litellm is None:
            raise RuntimeError("litellm is required for LLMClient operations") from _LITELLM_ERROR


    def _extract_content(self, response: Any) -> tuple[str, dict[str, Any]]:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is None and "tool_calls" in message:
                return json.dumps(message["tool_calls"]), {}
            reasoning_payload: dict[str, Any] = {}
            for key in ("reasoning_content", "thinking", "thinking_blocks"):
                value = message.get(key)
                if value:
                    reasoning_payload[key] = value
            usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
            if usage:
                details = None
                if isinstance(usage, dict):
                    details = usage.get("completion_tokens_details")
                else:
                    details = getattr(usage, "completion_tokens_details", None)
                if details:
                    tokens = getattr(details, "reasoning_tokens", None)
                    if not tokens and isinstance(details, dict):
                        tokens = details.get("reasoning_tokens")
                    if tokens:
                        reasoning_payload.setdefault("token_counts", {})["reasoning_tokens"] = tokens
            self._record_reasoning(reasoning_payload)
            if content is None or str(content).strip() == "":
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"LLM returned empty content. Full response: {response}")
            return str(content or ""), reasoning_payload
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format from LLM client. Response: {response}") from exc

    def _record_reasoning(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        try:
            from atlas.runtime.orchestration.execution_context import ExecutionContext
            context = ExecutionContext.get()
        except Exception:  # pragma: no cover - context not initialised
            return
        origin = context.metadata.pop("_reasoning_origin", None)
        if origin is None:
            return
        queue = context.metadata.setdefault("_llm_reasoning_queue", [])
        queue.append({"origin": origin, "payload": payload})

    def _mock_response(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> LLMResponse:
        user_content = ""
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                user_content = str(last_message.get("content", ""))
        if response_format and response_format.get("type") == "json_object":
            # Check if this looks like a learning synthesis request
            if "playbook_entry" in user_content or "learning" in user_content.lower():
                # Return mock playbook_entry.v1 structure
                payload = {
                    "version": "playbook_entry.v1",
                    "student_pamphlet": None,
                    "teacher_pamphlet": None,
                    "playbook_entries": [],
                    "session_student_learning": None,
                    "session_teacher_learning": None,
                    "metadata": None
                }
            elif "plan" in user_content:
                payload = {
                    "steps": [
                        {
                            "id": 1,
                            "description": "Mock step generated by ATLAS_OFFLINE_MODE.",
                            "depends_on": [],
                            "tool": None,
                            "tool_params": None,
                        }
                    ]
                }
            elif "guidance" in user_content:
                payload = {"valid": True, "guidance": "No additional guidance required."}
            else:
                payload = {"notes": "Mock JSON response from ATLAS_OFFLINE_MODE."}
            content = json.dumps(payload)
        else:
            content = "Mock response from ATLAS_OFFLINE_MODE."
        return LLMResponse(content=content, reasoning={}, raw={"mock": True})
