"""Python-callable tool bridge for GDPval references."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    from litellm import acompletion
    _LITELLM_AVAILABLE = True
except ImportError:
    acompletion = None
    _LITELLM_AVAILABLE = False

from .loader import CACHE_ROOT
from .loader import GDPValTask
from .loader import ReferenceNotFound
from .loader import load_manifest
from .loader import load_gdpval_tasks
from .loader import ensure_manifest


class GDPValAgentError(RuntimeError):
    """Raised when the GDPval agent encounters an error."""


@dataclass
class GDPValToolset:
    cache_root: Path = CACHE_ROOT

    def list_references(self, task_id: str) -> List[Dict[str, Any]]:
        manifest = load_manifest(task_id)
        references = manifest.get("references", [])
        return [
            {
                "filename": item.get("filename"),
                "media_type": item.get("media_type"),
                "cached_path": item.get("cached_path"),
                "text_path": item.get("text_path"),
                "source_url": item.get("source_url"),
            }
            for item in references
        ]

    def read_reference(self, task_id: str, filename: str) -> str:
        manifest = load_manifest(task_id)
        for entry in manifest.get("references", []):
            if entry.get("filename") == filename:
                text_path = entry.get("text_path")
                if not text_path:
                    raise ReferenceNotFound(f"Reference {filename} is not cached for task {task_id}")
                path = Path(text_path)
                if not path.exists():
                    raise ReferenceNotFound(f"Reference text missing at {text_path}")
                return path.read_text()
        raise ReferenceNotFound(f"Reference {filename} not found for task {task_id}")

    def summarize_reference(self, task_id: str, filename: str, max_lines: int = 20) -> str:
        text = self.read_reference(task_id, filename)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        excerpt = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            excerpt += "\n..."
        return excerpt

    def ensure_cache(self, task_id: str) -> None:
        cache_dir = self.cache_root / task_id
        if cache_dir.exists():
            return
        tasks = load_gdpval_tasks(split="train", cache_references=True)
        for task in tasks:
            if task.task_id == task_id:
                for reference in task.references:
                    reference.cache(task_id)
                ensure_manifest(task)
                return
        raise ReferenceNotFound(f"Task {task_id} not found in GDPval dataset")


_TOOLSET = GDPValToolset()


async def create_gdpval_agent(prompt: str, metadata: Dict[str, Any]) -> Any:
    """Handle tool invocations and general prompts for GDPval."""
    if not prompt or not prompt.strip():
        return {"error": "Empty prompt received"}

    # Check if this is a tool call
    try:
        payload = json.loads(prompt)
        if isinstance(payload, dict) and "tool" in payload:
            tool_info = payload["tool"]
            tool_name = tool_info.get("name")
            tool_args = tool_info.get("arguments", {})

            if tool_name == "list_references":
                return _TOOLSET.list_references(**tool_args)
            elif tool_name == "read_reference":
                return _TOOLSET.read_reference(**tool_args)
            elif tool_name == "summarize_reference":
                return _TOOLSET.summarize_reference(**tool_args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
    except (json.JSONDecodeError, ValueError):
        pass

    # Not a tool call - use LLM for reasoning
    llm_config = metadata.get("llm_config")
    if not llm_config:
        return {"error": "No LLM configuration provided"}

    if not _LITELLM_AVAILABLE:
        return {"error": "litellm is required but not installed"}

    # Prepare LLM call
    model = llm_config.get("model")
    api_key_env = llm_config.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm_config.get("temperature", 1.0),
        "max_tokens": llm_config.get("max_output_tokens", 32000),
    }

    if api_key:
        kwargs["api_key"] = api_key

    if llm_config.get("additional_headers"):
        kwargs["extra_headers"] = llm_config["additional_headers"]

    # Call LLM
    try:
        response = await acompletion(**kwargs)
        content = response.choices[0].message.content

        # Try to parse as JSON for planning mode
        mode = metadata.get("mode")
        if mode == "planning":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        return content
    except Exception as exc:
        return {"error": f"LLM call failed: {exc}"}


def build_session_metadata(task: GDPValTask) -> Dict[str, Any]:
    manifest = task.to_manifest()
    return {
        "task_id": task.task_id,
        "sector": task.sector,
        "occupation": task.occupation,
        "references": manifest.get("references", []),
    }


def export_manifest(task: GDPValTask) -> None:
    cache_dir = CACHE_ROOT / task.task_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        return
    manifest_path.write_text(json.dumps(task.to_manifest(), indent=2))
