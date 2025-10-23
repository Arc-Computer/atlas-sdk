"""Static analysis helpers for autodiscovering Atlas environments and agents."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}

Role = Literal["environment", "agent"]


@dataclass(slots=True)
class Candidate:
    role: Role
    module: str
    qualname: str
    file_path: Path
    score: int
    reason: str
    via_decorator: bool
    capabilities: dict[str, bool] = field(default_factory=dict)

    def dotted_path(self) -> str:
        return f"{self.module}:{self.qualname}"


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & _SKIP_DIRS:
            continue
        yield path


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    stem_parts = rel.with_suffix("").parts
    if stem_parts[-1] == "__init__":
        stem_parts = stem_parts[:-1]
    return ".".join(stem_parts)


def _has_method(node: ast.ClassDef, name: str) -> bool:
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return True
    return False


def _decorator_matches(node: ast.ClassDef, attr_name: str) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == attr_name:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == attr_name:
            return True
    return False


def _score_class(node: ast.ClassDef) -> tuple[Role | None, int, bool, dict[str, bool]]:
    capabilities: dict[str, bool] = {}
    if _decorator_matches(node, "environment"):
        capabilities.update({"decorated": True, "reset": True, "step": True, "close": True})
        return "environment", 120, True, capabilities
    if _decorator_matches(node, "agent"):
        capabilities.update({"decorated": True, "plan": True, "act": True, "summarize": True})
        return "agent", 120, True, capabilities

    env_caps = {
        "reset": _has_method(node, "reset"),
        "step": _has_method(node, "step"),
        "close": _has_method(node, "close"),
        "render": _has_method(node, "render"),
    }
    agent_caps = {
        "plan": _has_method(node, "plan"),
        "act": _has_method(node, "act"),
        "summarize": _has_method(node, "summarize"),
        "reset": _has_method(node, "reset"),
    }

    def _base_score(caps: dict[str, bool]) -> int:
        return sum(20 for value in caps.values() if value)

    if env_caps["reset"] and env_caps["step"]:
        env_caps["heuristic"] = True
        score = 80 + _base_score(env_caps)
        if any(isinstance(base, ast.Name) and base.id.lower() in {"env", "environment"} for base in node.bases):
            env_caps["gym_base"] = True
            score += 10
        return "environment", score, False, env_caps
    if agent_caps["act"]:
        agent_caps["heuristic"] = True
        score = 60 + _base_score(agent_caps)
        if any(isinstance(base, ast.Name) and "agent" in base.id.lower() for base in node.bases):
            agent_caps["agent_base"] = True
            score += 10
        return "agent", score, False, agent_caps
    return None, 0, False, capabilities


def discover_candidates(root: Path) -> list[Candidate]:
    root = root.resolve()
    candidates: list[Candidate] = []
    for path in _iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        module_name = _module_name(root, path)
        if not module_name:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            role, score, via_decorator, capabilities = _score_class(node)
            if role is None:
                continue
            reason = "decorator" if via_decorator else "heuristic"
            candidates.append(
                Candidate(
                    role=role,
                    module=module_name,
                    qualname=node.name,
                    file_path=path,
                    score=score,
                    reason=reason,
                    via_decorator=via_decorator,
                    capabilities=capabilities or {},
                )
            )
    candidates.sort(key=lambda cand: (cand.role, -cand.score, cand.module, cand.qualname))
    return candidates


def split_candidates(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    envs = [cand for cand in candidates if cand.role == "environment"]
    agents = [cand for cand in candidates if cand.role == "agent"]
    return envs, agents


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_candidate(candidate: Candidate, project_root: Path) -> dict[str, object]:
    rel_path = candidate.file_path.resolve().relative_to(project_root.resolve())
    return {
        "role": candidate.role,
        "module": candidate.module,
        "qualname": candidate.qualname,
        "file": str(rel_path),
        "hash": calculate_file_hash(candidate.file_path),
        "score": candidate.score,
        "reason": candidate.reason,
    }


def write_discovery_payload(
    destination: Path,
    *,
    metadata: dict[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
