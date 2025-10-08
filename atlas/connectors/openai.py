"""OpenAI-compatible adapter implemented with litellm."""

from __future__ import annotations

import json
import os
from typing import Any
from typing import Dict
from typing import List

try:
    import litellm
    from litellm import acompletion
    _LITELLM_ERROR = None
except ModuleNotFoundError as exc:
    litellm = None
    acompletion = None
    _LITELLM_ERROR = exc

from atlas.connectors.registry import AdapterError
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import register_adapter
from atlas.config.models import AdapterType
from atlas.config.models import OpenAIAdapterConfig


class OpenAIAdapter(AgentAdapter):
    """Adapter that proxies chat completions to OpenAI compatible endpoints."""

    def __init__(self, config: OpenAIAdapterConfig):
        self._config = config

    def _build_messages(self, prompt: str, metadata: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._config.system_prompt:
            messages.append({"role": "system", "content": self._config.system_prompt})
        entries = metadata.get("messages") if metadata else None
        if entries:
            for entry in entries:
                converted = self._convert_metadata_entry(entry)
                if converted:
                    messages.append(converted)
        elif metadata:
            messages.append({"role": "system", "content": json.dumps(metadata)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _convert_metadata_entry(self, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        role = entry.get("role")
        if not role:
            role = self._map_entry_type(entry.get("type"))
        if not role:
            return None
        message: Dict[str, Any] = {"role": role, "content": self._stringify_content(entry.get("content"))}
        if role == "assistant":
            tool_calls = self._normalise_tool_calls(entry.get("tool_calls"))
            if tool_calls:
                message["tool_calls"] = tool_calls
        if role == "tool" and entry.get("tool_call_id"):
            message["tool_call_id"] = entry["tool_call_id"]
        return message

    def _map_entry_type(self, entry_type: str | None) -> str | None:
        mapping = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
        }
        return mapping.get(entry_type or "")

    def _normalise_tool_calls(self, raw_tool_calls: Any) -> List[Dict[str, Any]]:
        if raw_tool_calls is None:
            return []
        if isinstance(raw_tool_calls, str):
            try:
                raw_tool_calls = json.loads(raw_tool_calls)
            except json.JSONDecodeError:
                return []
        if isinstance(raw_tool_calls, dict):
            raw_tool_calls = [raw_tool_calls]
        tool_calls: List[Dict[str, Any]] = []
        for item in raw_tool_calls:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    continue
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            arguments = item.get("arguments") or item.get("args") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            cleaned: Dict[str, Any] = {"name": name, "arguments": arguments}
            if item.get("id"):
                cleaned["id"] = item["id"]
            if item.get("type"):
                cleaned["type"] = item["type"]
            tool_calls.append(cleaned)
        return tool_calls

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return str(content)

    def _base_kwargs(self) -> Dict[str, Any]:
        llm = self._config.llm
        api_key = os.getenv(llm.api_key_env)
        if not api_key:
            raise AdapterError(f"environment variable '{llm.api_key_env}' is not set")
        kwargs: Dict[str, Any] = {
            "model": llm.model,
            "api_key": api_key,
            "temperature": llm.temperature,
            "timeout": llm.timeout_seconds,
        }
        if llm.api_base:
            kwargs["api_base"] = llm.api_base
        if llm.organization:
            kwargs["organization"] = llm.organization
        if llm.top_p is not None:
            kwargs["top_p"] = llm.top_p
        if llm.max_output_tokens is not None:
            kwargs["max_tokens"] = llm.max_output_tokens
        if llm.additional_headers:
            kwargs["extra_headers"] = llm.additional_headers
        if self._config.response_format:
            kwargs["response_format"] = self._config.response_format
        supports_reasoning = False
        if litellm is not None and hasattr(litellm, "supports_reasoning"):
            try:
                supports_reasoning = bool(litellm.supports_reasoning(llm.model))
            except Exception:
                supports_reasoning = False
        if supports_reasoning:
            headers = dict(kwargs.get("extra_headers") or {})
            headers.setdefault("OpenAI-Beta", "reasoning=1")
            kwargs["extra_headers"] = headers
            kwargs["temperature"] = 1.0
            if llm.reasoning_effort:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body.setdefault("reasoning_effort", llm.reasoning_effort)
                kwargs["extra_body"] = extra_body
        return kwargs

    def _parse_response(self, response: Any) -> str:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is None and "tool_calls" in message:
                return json.dumps(message["tool_calls"])
            return str(content or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("unexpected response format from OpenAI adapter") from exc

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        if acompletion is None:
            raise AdapterError("litellm is required for OpenAIAdapter") from _LITELLM_ERROR
        messages = self._build_messages(prompt, metadata)
        kwargs = self._base_kwargs()
        kwargs["messages"] = messages
        try:
            response = await acompletion(**kwargs)
        except Exception as exc:
            raise AdapterError("openai adapter request failed") from exc
        return self._parse_response(response)


register_adapter(AdapterType.OPENAI, OpenAIAdapter)

__all__ = ["OpenAIAdapter"]
