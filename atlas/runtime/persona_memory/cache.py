"""In-process cache for persona memory lookups."""

from __future__ import annotations

import asyncio
import os
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from atlas.config.models import AtlasConfig

if typing.TYPE_CHECKING:
    from atlas.runtime.storage.database import Database

Truthies = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PersonaMemoryKey:
    """Cache key for persona memory lookups."""

    agent_name: str
    tenant_id: str
    fingerprint: str
    persona: str


class PersonaMemoryCache:
    """Caches persona memories to limit duplicate database fetches."""

    def __init__(self) -> None:
        self._cache: Dict[PersonaMemoryKey, List[Dict[str, Any]]] = {}
        self._locks: Dict[PersonaMemoryKey, asyncio.Lock] = {}

    async def get_or_load(
        self,
        database: "Database",
        key: PersonaMemoryKey,
        statuses: Sequence[str] | None,
        *,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return persona memories, loading them once when caching is enabled."""
        if not use_cache:
            return await database.fetch_persona_memories(
                key.agent_name,
                key.tenant_id,
                key.persona,
                key.fingerprint,
                statuses=statuses,
            )
        if key in self._cache:
            return self._cache[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._cache:
                return self._cache[key]
            records = await database.fetch_persona_memories(
                key.agent_name,
                key.tenant_id,
                key.persona,
                key.fingerprint,
                statuses=statuses,
            )
            self._cache[key] = records
            return records

    def invalidate(self, key: PersonaMemoryKey) -> None:
        """Remove a cached persona memory entry."""
        self._cache.pop(key, None)
        self._locks.pop(key, None)

    def clear(self) -> None:
        """Clear all cached persona memories."""
        self._cache.clear()
        self._locks.clear()


_CACHE: PersonaMemoryCache | None = None


def get_cache() -> PersonaMemoryCache:
    """Return the process-wide persona memory cache."""
    global _CACHE
    if _CACHE is None:
        _CACHE = PersonaMemoryCache()
    return _CACHE


def is_cache_disabled(config: AtlasConfig | None = None) -> bool:
    """Determine whether persona memory caching is disabled."""
    env_value = os.getenv("ATLAS_PERSONA_MEMORY_CACHE_DISABLED")
    if env_value and env_value.strip().lower() in Truthies:
        return True
    if config:
        persona_cfg = config.metadata.get("persona_memory") if getattr(config, "metadata", None) else None
        if isinstance(persona_cfg, dict):
            flag = persona_cfg.get("cache_disabled")
            if isinstance(flag, str):
                return flag.strip().lower() in Truthies
            if flag:
                return True
    return False
