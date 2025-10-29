"""Static analysis helpers for autodiscovering Atlas environments and agents."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
import re
import subprocess
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
    signals: dict[str, object] = field(default_factory=dict)

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


def _method_raises_not_implemented(node: ast.ClassDef, method_name: str) -> bool:
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
            for stmt in child.body:
                if isinstance(stmt, ast.Raise):
                    exc = stmt.exc
                    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                        return True
                    if isinstance(exc, ast.Call):
                        func = exc.func
                        if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                            return True
                        if isinstance(func, ast.Attribute) and func.attr == "NotImplementedError":
                            return True
    return False


def _regex_extract_strings(pattern: re.Pattern[str], source: str) -> list[str]:
    result: list[str] = []
    for match in pattern.finditer(source):
        literal = match.group("value")
        try:
            value = ast.literal_eval(literal)
        except Exception:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                result.append(text)
    return result


def _regex_extract_literals(pattern: re.Pattern[str], source: str) -> list[object]:
    result: list[object] = []
    for match in pattern.finditer(source):
        literal = match.group("value")
        try:
            value = ast.literal_eval(literal)
        except Exception:
            continue
        result.append(value)
    return result


def _find_factory_hits(source: str, qualname: str) -> list[str]:
    pattern = re.compile(
        r"def\s+(create_[a-zA-Z0-9_]+)\s*\([^)]*\)\s*:\s*\n\s+return\s+(?:[A-Za-z0-9_]+\()?"
        + re.escape(qualname)
    )
    return [match.group(1) for match in pattern.finditer(source)]


def _run_ripgrep(root: Path, pattern: str) -> int:
    try:
        result = subprocess.run(
            [
                "rg",
                "--no-heading",
                "--color",
                "never",
                "--max-count",
                "5",
                pattern,
                str(root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0
    if result.returncode not in {0, 1}:
        return 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def _collect_candidate_signals(
    root: Path,
    source: str,
    node: ast.ClassDef,
    module_name: str,
    role: Role,
) -> dict[str, object]:
    signals: dict[str, object] = {}
    abstract_methods: list[str] = []
    method_names = ["act", "plan", "summarize"] if role == "agent" else ["reset", "step", "close"]
    for method_name in method_names:
        if _method_raises_not_implemented(node, method_name):
            abstract_methods.append(method_name)
    signals["abstract_methods"] = abstract_methods
    prompt_pattern = re.compile(
        r"^(?P<name>[A-Z0-9_]*PROMPT[A-Z0-9_]*)\s*=\s*(?P<value>(?:\"\"\".*?\"\"\"|'''.*?'''|\".*?\"|'.*?'))",
        re.MULTILINE | re.DOTALL,
    )
    config_pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_]*config[A-Za-z0-9_]*)\s*=\s*(?P<value>(?:\[[\s\S]*?\]|{[\s\S]*?}))",
        re.IGNORECASE | re.MULTILINE,
    )
    tool_pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_]*tool[A-Za-z0-9_]*)\s*=\s*(?P<value>(?:\[[\s\S]*?\]|{[\s\S]*?}))",
        re.IGNORECASE | re.MULTILINE,
    )
    signals["prompt_literals"] = _regex_extract_strings(prompt_pattern, source)
    signals["config_literals"] = _regex_extract_literals(config_pattern, source)
    signals["tool_literals"] = _regex_extract_literals(tool_pattern, source)
    signals["factory_functions"] = _find_factory_hits(source, node.name)
    instantiation_pattern = rf"{re.escape(node.name)}\s*\("
    signals["instantiations"] = _run_ripgrep(root, instantiation_pattern)
    import_pattern = rf"from\s+{re.escape(module_name)}\s+import\s+{re.escape(node.name)}"
    signals["import_hits"] = _run_ripgrep(root, import_pattern)
    if role == "environment":
        keywords = ["mysql", "docker", "dataset", "table", "reward", "observation"]
        signals["environment_keywords"] = sum(source.lower().count(keyword) for keyword in keywords)
    return signals


def _compute_signal_adjustment(role: Role, signals: dict[str, object]) -> int:
    adjustment = 0
    abstract_methods = signals.get("abstract_methods") or []
    if abstract_methods:
        adjustment -= 70 * len(abstract_methods)
    if signals.get("prompt_literals"):
        adjustment += 35
    if signals.get("config_literals"):
        adjustment += 25
    if signals.get("tool_literals"):
        adjustment += 20
    if signals.get("factory_functions"):
        adjustment += 25
    instantiations = signals.get("instantiations") or 0
    import_hits = signals.get("import_hits") or 0
    usage_hits = instantiations + import_hits
    if usage_hits:
        adjustment += min(50, usage_hits * 10)
    else:
        adjustment -= 20
    if role == "environment":
        keyword_hits = signals.get("environment_keywords") or 0
        if keyword_hits:
            adjustment += min(30, keyword_hits * 2)
    return adjustment


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
            signals = _collect_candidate_signals(root, source, node, module_name, role)
            score += _compute_signal_adjustment(role, signals)
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
                    signals=signals,
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
        "signals": {
            "instantiations": candidate.signals.get("instantiations"),
            "abstract_methods": candidate.signals.get("abstract_methods"),
            "prompt_literals": len(candidate.signals.get("prompt_literals") or []),
            "config_literals": len(candidate.signals.get("config_literals") or []),
        },
    }


def write_discovery_payload(
    destination: Path,
    *,
    metadata: dict[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _find_class_definition(tree: ast.Module, qualname: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == qualname:
            return node
    return None


def _format_default(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    try:
        return ast.unparse(value).strip()
    except Exception:
        return None


def _extract_parameters(class_node: ast.ClassDef) -> list[dict[str, object]]:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            parameters: list[dict[str, object]] = []
            args = node.args
            positional = args.args[1:] if args.args else []
            defaults = list(args.defaults) if args.defaults else []
            default_offset = len(positional) - len(defaults)
            for index, arg in enumerate(positional):
                default_index = index - default_offset
                default_value = defaults[default_index] if default_index >= 0 else None
                parameters.append(
                    {
                        "name": arg.arg,
                        "default": _format_default(default_value),
                        "required": default_value is None,
                    }
                )
            if args.vararg:
                parameters.append(
                    {
                        "name": f"*{args.vararg.arg}",
                        "default": None,
                        "required": False,
                    }
                )
            for kw_index, kw_arg in enumerate(args.kwonlyargs or []):
                default_value = args.kw_defaults[kw_index] if args.kw_defaults else None
                parameters.append(
                    {
                        "name": kw_arg.arg,
                        "default": _format_default(default_value),
                        "required": default_value is None,
                    }
                )
            if args.kwarg:
                parameters.append(
                    {
                        "name": f"**{args.kwarg.arg}",
                        "default": None,
                        "required": False,
                    }
                )
            return parameters
    return []


def _normalise_tools(tool_literals: list[object]) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    for entry in tool_literals or []:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("id")
            tool_data: dict[str, object] = {}
            if name:
                tool_data["name"] = str(name)
            for key in ("description", "type"):
                if key in entry:
                    tool_data[key] = entry[key]
            if tool_data:
                tools.append(tool_data)
        elif isinstance(entry, list):
            for item in entry:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id")
                    if name:
                        tools.append({"name": str(name)})
                elif isinstance(item, str):
                    tools.append({"name": item})
        elif isinstance(entry, str):
            tools.append({"name": entry})
    return tools


def collect_runtime_metadata(project_root: Path, candidate: Candidate | None) -> dict[str, object]:
    if candidate is None:
        return {}
    try:
        source = candidate.file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(candidate.file_path))
    except Exception:
        return {}
    class_node = _find_class_definition(tree, candidate.qualname)
    if class_node is None:
        return {}
    metadata: dict[str, object] = {
        "module": candidate.module,
        "qualname": candidate.qualname,
        "prompts": list(candidate.signals.get("prompt_literals") or []),
        "tools": _normalise_tools(candidate.signals.get("tool_literals") or []),
        "config_literals": candidate.signals.get("config_literals") or [],
        "factory_functions": candidate.signals.get("factory_functions") or [],
        "instantiations": candidate.signals.get("instantiations"),
        "import_hits": candidate.signals.get("import_hits"),
    }
    metadata["parameters"] = _extract_parameters(class_node)
    return metadata
