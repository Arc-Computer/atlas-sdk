"""Static analysis helpers for autodiscovering Atlas environments and agents."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Sequence

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


def _score_class(node: ast.ClassDef) -> tuple[Role | None, int, bool]:
    if _decorator_matches(node, "environment"):
        return "environment", 100, True
    if _decorator_matches(node, "agent"):
        return "agent", 100, True
    env_methods = {"reset", "step", "close"}
    if all(_has_method(node, method) for method in env_methods):
        return "environment", 60, False
    agent_methods = {"plan", "act", "summarize"}
    if all(_has_method(node, method) for method in agent_methods):
        return "agent", 60, False
    return None, 0, False


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
            role, score, via_decorator = _score_class(node)
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
