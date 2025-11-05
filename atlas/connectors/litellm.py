"""Multi-provider adapter implemented with litellm."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

try:
    import litellm  # type: ignore[import-untyped]
    from litellm import acompletion  # type: ignore[import-untyped]
    _LITELLM_ERROR = None
except ModuleNotFoundError as exc:
    litellm = None  # type: ignore[assignment]
    acompletion = None  # type: ignore[assignment]
    _LITELLM_ERROR = exc

import warnings

from atlas.connectors.prompt_digest import build_prompt_digest, PromptDigestTooLargeError
from atlas.connectors.registry import AdapterError
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import register_adapter
from atlas.connectors.utils import AdapterResponse, normalise_usage_payload
from atlas.config.models import AdapterType, AdapterUnion, LitellmAdapterConfig, OpenAIAdapterConfig
from atlas.runtime.orchestration.execution_context import ExecutionContext


logger = logging.getLogger(__name__)


class LitellmAdapter(AgentAdapter):
    """Multi-provider adapter proxying chat completions via litellm."""

    def __init__(self, config: LitellmAdapterConfig):
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
            try:
                digest = build_prompt_digest(metadata, self._config.llm, self._config.metadata_digest)
            except PromptDigestTooLargeError as exc:
                raise AdapterError(str(exc)) from exc
            messages.append({"role": "system", "content": digest})
            try:
                stats = json.loads(digest).get("digest_stats", {})
            except json.JSONDecodeError:
                stats = {}
            if stats.get("omitted_sections"):
                logger.debug("metadata digest omitted sections: %s", stats["omitted_sections"])
            # Store digest_stats in ExecutionContext metadata for artifact capture
            if stats:
                try:
                    context = ExecutionContext.get()
                    context.metadata["digest_stats"] = stats
                except Exception:  # pragma: no cover - instrumentation best effort
                    logger.debug("Failed to store digest_stats", exc_info=True)
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

    def _parse_response(self, response: Any) -> AdapterResponse:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            tool_calls_raw = message.get("tool_calls")
            if tool_calls_raw:
                logger.debug("Tool calls detected in LLM response: %d calls", len(tool_calls_raw) if isinstance(tool_calls_raw, list) else 1)
            tool_calls = self._normalise_tool_calls(tool_calls_raw) if tool_calls_raw is not None else None
            normalised_content = self._stringify_content(content) if content is not None else ""
            raw_usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
            usage = normalise_usage_payload(raw_usage)
            return AdapterResponse(normalised_content, tool_calls=tool_calls, usage=usage)
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("unexpected response format from litellm adapter") from exc

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> AdapterResponse:
        if acompletion is None:
            raise AdapterError("litellm is required for LitellmAdapter") from _LITELLM_ERROR
        messages = self._build_messages(prompt, metadata)
        kwargs = self._base_kwargs()
        kwargs["messages"] = messages
        # Extract tools and tool_choice from metadata and pass to litellm
        tools = metadata.get("tools") if metadata else None
        tool_choice = metadata.get("tool_choice") if metadata else None
        if tools:
            kwargs["tools"] = tools
            logger.debug("Passing %d tools to litellm: %s", len(tools), [t.get("function", {}).get("name", "unknown") for t in tools])
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                logger.debug("Tool choice set to: %s", tool_choice)
            else:
                logger.debug("No tool_choice specified, using default")
        else:
            logger.debug("No tools found in metadata")
        try:
            response = await acompletion(**kwargs)
        except Exception as exc:
            raise AdapterError("litellm adapter request failed") from exc
        return self._parse_response(response)


def _build_litellm_adapter(config: AdapterUnion) -> AgentAdapter:
    if not isinstance(config, LitellmAdapterConfig):
        raise AdapterError("Litellm adapter requires LitellmAdapterConfig")
    return LitellmAdapter(config)


def _build_openai_adapter_deprecated(config: AdapterUnion) -> AgentAdapter:
    warnings.warn(
        "AdapterType.OPENAI is deprecated. Use AdapterType.LITELLM with type: litellm instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not isinstance(config, OpenAIAdapterConfig):
        raise AdapterError("OpenAI adapter requires OpenAIAdapterConfig")
    # Convert to LitellmAdapterConfig for processing
    litellm_config = LitellmAdapterConfig(
        type=AdapterType.LITELLM,
        name=config.name,
        system_prompt=config.system_prompt,
        tools=config.tools,
        llm=config.llm,
        response_format=config.response_format,
        metadata_digest=config.metadata_digest,
    )
    return LitellmAdapter(litellm_config)


register_adapter(AdapterType.LITELLM, _build_litellm_adapter)
register_adapter(AdapterType.OPENAI, _build_openai_adapter_deprecated)

__all__ = ["LitellmAdapter"]

