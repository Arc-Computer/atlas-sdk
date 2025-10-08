"""Async wrapper around OpenAI's Responses API for the ARC demo."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from openai import AsyncOpenAI


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the ARC demo agent.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def invoke(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Send the user prompt through OpenAI Responses."""
    client = _client()
    model = os.getenv("ATLAS_ARC_DEMO_MODEL", "o4-mini")
    reasoning_effort = os.getenv("ATLAS_ARC_DEMO_REASONING", "medium")
    metadata = metadata or {}
    response = await client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        metadata={"source": "arc-agi-demo", **metadata},
    )
    text = getattr(response, "output_text", None)
    if not text and response.output and response.output[0].content:
        pieces = []
        for item in response.output:
            for block in item.content:
                value = getattr(block, "text", None) or getattr(block, "content", "")
                if isinstance(value, str):
                    pieces.append(value)
        text = "\n".join(pieces)
    if not text:
        raise RuntimeError("OpenAI response did not contain text output.")
    return text.strip()
