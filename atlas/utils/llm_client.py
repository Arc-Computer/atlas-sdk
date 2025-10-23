"""Lightweight wrapper around litellm for synchronous and asynchronous calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from typing import Dict
from typing import Sequence

try:
    import litellm  # type: ignore[import-untyped]
    _LITELLM_ERROR = None
    litellm.drop_params = True
except ModuleNotFoundError as exc:
    litellm = None  # type: ignore[assignment]
    _LITELLM_ERROR = exc

from atlas.config.models import LLMParameters


@dataclass
class LLMResponse:
    content: str
    raw: Any
    reasoning: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, parameters: LLMParameters) -> None:
        self._params = parameters
        self._mock_mode = os.getenv("ATLAS_FAKE_LLM", "0") not in {"0", "", "false", "False"}

    @property
    def model(self) -> str:
        return self._params.model

    async def acomplete(
        self,
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None = None,
        overrides: Dict[str, Any] | None = None,
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
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._mock_mode:
            return self._mock_response(messages, response_format)
        self._ensure_client()
        kwargs = self._prepare_kwargs(messages, response_format, overrides)
        result = litellm.completion(**kwargs)
        content, reasoning = self._extract_content(result)
        return LLMResponse(content=content, reasoning=reasoning, raw=result)

    def _prepare_kwargs(
        self,
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None,
        overrides: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        params = self._params
        api_key = os.getenv(params.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable '{params.api_key_env}' is not set")
        overrides = overrides or {}
        kwargs: Dict[str, Any] = {"model": params.model, "messages": list(messages), "api_key": api_key}
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
        if response_format:
            kwargs["response_format"] = response_format

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


    def _extract_content(self, response: Any) -> tuple[str, Dict[str, Any]]:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is None and "tool_calls" in message:
                return json.dumps(message["tool_calls"]), {}
            reasoning_payload: Dict[str, Any] = {}
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

    def _record_reasoning(self, payload: Dict[str, Any]) -> None:
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
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None,
    ) -> LLMResponse:
        user_content = ""
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                user_content = str(last_message.get("content", ""))
        if response_format and response_format.get("type") == "json_object":
            if "plan" in user_content:
                payload = {
                    "steps": [
                        {
                            "id": 1,
                            "description": "Mock step generated by ATLAS_FAKE_LLM.",
                            "depends_on": [],
                            "tool": None,
                            "tool_params": None,
                        }
                    ]
                }
            elif "guidance" in user_content:
                payload = {"valid": True, "guidance": "No additional guidance required."}
            else:
                payload = {"notes": "Mock JSON response from ATLAS_FAKE_LLM."}
            content = json.dumps(payload)
        else:
            content = "Mock response from ATLAS_FAKE_LLM."
        return LLMResponse(content=content, reasoning={}, raw={"mock": True})
