"""Lightweight wrapper around litellm for synchronous and asynchronous calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

try:
    from litellm import acompletion
    from litellm import completion
    _LITELLM_ERROR = None
except ModuleNotFoundError as exc:
    acompletion = None
    completion = None
    _LITELLM_ERROR = exc

from atlas.config.models import LLMParameters


@dataclass
class LLMResponse:
    content: str
    raw: Any


class LLMClient:
    def __init__(self, parameters: LLMParameters) -> None:
        self._params = parameters

    async def acomplete(
        self,
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._ensure_client()
        kwargs = self._prepare_kwargs(messages, response_format, overrides)
        result = await acompletion(**kwargs)
        return LLMResponse(content=self._extract_content(result), raw=result)

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        response_format: Dict[str, Any] | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._ensure_client()
        kwargs = self._prepare_kwargs(messages, response_format, overrides)
        result = completion(**kwargs)
        return LLMResponse(content=self._extract_content(result), raw=result)

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
        kwargs: Dict[str, Any] = {
            "model": params.model,
            "messages": list(messages),
            "api_key": api_key,
            "temperature": params.temperature,
            "timeout": params.timeout_seconds,
        }
        if params.api_base:
            kwargs["api_base"] = params.api_base
        if params.organization:
            kwargs["organization"] = params.organization
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        if params.additional_headers:
            kwargs["extra_headers"] = params.additional_headers
        if response_format:
            kwargs["response_format"] = response_format
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    kwargs[key] = value
        return kwargs

    def _ensure_client(self) -> None:
        if acompletion is None or completion is None:
            raise RuntimeError("litellm is required for LLMClient operations") from _LITELLM_ERROR


    def _extract_content(self, response: Any) -> str:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is None and "tool_calls" in message:
                return json.dumps(message["tool_calls"])
            return str(content or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected response format from LLM client") from exc
