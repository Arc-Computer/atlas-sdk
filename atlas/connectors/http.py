"""HTTP adapter for executing remote BYOA agents."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any
from typing import Dict

import httpx

from atlas.connectors.registry import AdapterError
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import register_adapter
from atlas.config.models import AdapterType
from atlas.config.models import HTTPAdapterConfig

class HTTPAdapter(AgentAdapter):
    """Adapter that exchanges JSON payloads over HTTP."""

    def __init__(self, config: HTTPAdapterConfig):
        self._config = config

    def _build_payload(self, prompt: str, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = copy.deepcopy(self._config.payload_template)
        payload.setdefault("prompt", prompt)
        if metadata:
            payload.setdefault("metadata", metadata)
        return payload

    def _extract_result(self, data: Any) -> str:
        path = list(self._config.result_path or ["output"])
        node = data
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                raise AdapterError(f"response missing key '{key}' while traversing result path")
        if isinstance(node, (dict, list)):
            return json.dumps(node)
        return str(node)

    async def _request_once(self, payload: Dict[str, Any]) -> httpx.Response:
        transport = self._config.transport
        async with httpx.AsyncClient(base_url=transport.base_url,
                                     headers=transport.headers,
                                     timeout=transport.timeout_seconds) as client:
            response = await client.post("", json=payload)
        return response

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        payload = self._build_payload(prompt, metadata)
        retry = self._config.transport.retry
        last_error: Exception | None = None
        for attempt in range(1, retry.attempts + 1):
            try:
                response = await self._request_once(payload)
                response.raise_for_status()
                return self._extract_result(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == retry.attempts:
                    break
                await asyncio.sleep(retry.backoff_seconds * attempt)
        raise AdapterError("http adapter failed to obtain a response") from last_error

register_adapter(AdapterType.HTTP, HTTPAdapter)

__all__ = ["HTTPAdapter"]
