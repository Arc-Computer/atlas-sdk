"""Static analysis helpers for autodiscovering Atlas environments and agents."""

from __future__ import annotations

import ast
import hashlib
import json
try:
    import yaml  # type: ignore[import-untyped]
except Exception:
    yaml = None
try:  # pragma: no cover - tomllib missing on older interpreters
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]
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
    is_factory: bool = False
    factory_kind: str | None = None

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


def _extract_return_call_targets(function: ast.FunctionDef) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                targets.add(".".join(reversed(parts)))
            elif isinstance(func, ast.Name):
                targets.add(func.id)
    return targets


def _score_function_candidate(
    function: ast.FunctionDef,
    module_name: str,
    path: Path,
    root: Path,
    source: str,
) -> Candidate | None:
    name_lower = function.name.lower()
    role: Role | None = None
    if "env" in name_lower or "environment" in name_lower:
        role = "environment"
    if "agent" in name_lower:
        role = "agent" if role is None else role
    if role is None:
        return None
    call_targets = _extract_return_call_targets(function)
    base_score = 50 if call_targets else 30
    framework_keywords = {
        "langgraph",
        "deepagents",
        "langchain",
        "autogen",
        "crewai",
        "semantic_kernel",
        "semantic-kernel",
        "openai",
        "anthropic",
        "llama_index",
        "haystack",
        "milvus",
        "qdrant",
        "chromadb",
        "weaviate",
    }
    framework_hints = {
        keyword: any(keyword in target.lower() for target in call_targets)
        for keyword in framework_keywords
    }
    boost = sum(20 for hit in framework_hints.values() if hit)
    score = base_score + boost
    reason = "factory"
    signals: dict[str, object] = {
        "abstract_methods": [],
        "prompt_literals": [],
        "config_literals": [],
        "instantiations": _run_ripgrep(root, rf"{re.escape(function.name)}\s*\(") or None,
        "call_targets": sorted(call_targets),
        "framework_hints": [name for name, hit in framework_hints.items() if hit],
    }
    return Candidate(
        role=role,
        module=module_name,
        qualname=function.name,
        file_path=path,
        score=score,
        reason=reason,
        via_decorator=False,
        capabilities={},
        signals=signals,
        is_factory=True,
        factory_kind="callable",
    )


def _score_assignment_candidate(
    assignment: ast.AST,
    module_name: str,
    path: Path,
    root: Path,
    source: str,
) -> Candidate | None:
    if isinstance(assignment, ast.Assign):
        targets = assignment.targets
        value = assignment.value
    elif isinstance(assignment, ast.AnnAssign):
        targets = [assignment.target]
        value = assignment.value
    else:
        return None
    if not targets or not isinstance(targets[0], ast.Name):
        return None
    target_name = targets[0].id
    if not isinstance(value, ast.Call):
        return None
    call_targets = _extract_return_call_targets(ast.FunctionDef(name="__temp__", args=ast.arguments(), body=[ast.Return(value=value)]))
    role: Role | None = None
    lower_target = target_name.lower()
    if "env" in lower_target or "environment" in lower_target:
        role = "environment"
    if "agent" in lower_target:
        role = "agent" if role is None else role
    if role is None:
        if any("agent" in call for call in call_targets):
            role = "agent"
        elif any("env" in call or "environment" in call for call in call_targets):
            role = "environment"
    if role is None:
        return None
    base_score = 45
    framework_keywords = {
        "langgraph",
        "deepagents",
        "langchain",
        "autogen",
        "crewai",
        "semantic_kernel",
        "semantic-kernel",
        "openai",
        "anthropic",
        "llama_index",
        "haystack",
        "milvus",
        "qdrant",
        "chromadb",
        "weaviate",
    }
    framework_hints = {
        keyword: any(keyword in target.lower() for target in call_targets)
        for keyword in framework_keywords
    }
    score = base_score + sum(20 for hit in framework_hints.values() if hit)
    signals: dict[str, object] = {
        "call_targets": sorted(call_targets),
        "framework_hints": [name for name, hit in framework_hints.items() if hit],
        "instantiations": _run_ripgrep(root, rf"{re.escape(target_name)}\s*=") or None,
    }
    return Candidate(
        role=role,
        module=module_name,
        qualname=target_name,
        file_path=path,
        score=score,
        reason="factory",
        via_decorator=False,
        capabilities={},
        signals=signals,
        is_factory=True,
        factory_kind="attribute",
    )


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
            if isinstance(node, ast.ClassDef):
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
                        is_factory=False,
                    )
                )
            elif isinstance(node, ast.FunctionDef):
                function_candidate = _score_function_candidate(node, module_name, path, root, source)
                if function_candidate is not None:
                    candidates.append(function_candidate)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                assignment_candidate = _score_assignment_candidate(node, module_name, path, root, source)
                if assignment_candidate is not None:
                    candidates.append(assignment_candidate)
    entrypoint_candidates = _discover_entrypoint_factories(root)
    existing_paths = {(cand.module, cand.qualname) for cand in candidates}
    for candidate in entrypoint_candidates:
        key = (candidate.module, candidate.qualname)
        if key not in existing_paths:
            candidates.append(candidate)
    candidates.sort(key=lambda cand: (cand.role, -cand.score, cand.module, cand.qualname))
    return candidates


def _load_module_ast(root: Path, module: str) -> tuple[Path | None, ast.Module | None, str | None]:
    module_path = root.joinpath(*module.split("."))
    file_path = None
    if module_path.with_suffix(".py").exists():
        file_path = module_path.with_suffix(".py")
    elif module_path.exists() and module_path.is_dir():
        candidate = module_path / "__init__.py"
        if candidate.exists():
            file_path = candidate
    if file_path is None:
        return None, None, None
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        return file_path, tree, source
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None, None, None


def _iter_entrypoint_specs(pyproject: dict[str, Any]) -> dict[str, str]:
    scripts: dict[str, str] = {}
    project_section = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    if isinstance(project_section.get("scripts"), dict):
        scripts.update(project_section["scripts"])
    entry_points = project_section.get("entry-points")
    if isinstance(entry_points, dict):
        for entries in entry_points.values():
            if isinstance(entries, dict):
                scripts.update(entries)
    tool_section = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    poetry_section = tool_section.get("poetry") if isinstance(tool_section.get("poetry"), dict) else {}
    if isinstance(poetry_section.get("scripts"), dict):
        scripts.update(poetry_section["scripts"])
    return scripts


def _discover_entrypoint_factories(root: Path) -> list[Candidate]:
    if tomllib is None:
        return []
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    entrypoints = _iter_entrypoint_specs(data)
    candidates: list[Candidate] = []
    for target in entrypoints.values():
        if not isinstance(target, str) or ":" not in target:
            continue
        module, _, attr = target.partition(":")
        module = module.strip()
        attr = attr.strip()
        if not module or not attr:
            continue
        file_path, tree, source = _load_module_ast(root, module)
        if tree is None or source is None:
            continue
        attr_head = attr.split(".")[0]
        function_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == attr_head), None)
        if function_node is None:
            continue
        candidate = _score_function_candidate(function_node, module, file_path, root, source)
        if candidate is None:
            continue
        candidate.reason = "entrypoint"
        candidate.signals.setdefault("entrypoint", attr)
        candidate.is_factory = True
        candidates.append(candidate)
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
        "is_factory": candidate.is_factory,
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


def _find_function_definition(tree: ast.Module, qualname: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == qualname:
            return node
    return None


def _find_assignment_definition(tree: ast.Module, qualname: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == qualname:
                    return node
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == qualname:
                return node
    return None


def _format_default(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    try:
        return ast.unparse(value).strip()
    except Exception:
        return None


def _evaluate_literal(node: ast.AST | None, pool: dict[str, object]) -> object | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return pool.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                return None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items: list[object] = []
        for elt in node.elts:
            evaluated = _evaluate_literal(elt, pool)
            if evaluated is None:
                return None
            items.append(evaluated)
        return items
    return None


def _build_constant_pool(tree: ast.Module) -> dict[str, object]:
    pool: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _evaluate_literal(node.value, pool)
            if value is not None:
                pool[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _evaluate_literal(node.value, pool)
            if value is not None:
                pool[node.target.id] = value
    return pool


def _extract_tool_names(expr: ast.AST, pool: dict[str, object]) -> list[str]:
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        names: list[str] = []
        for elt in expr.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(ast.unparse(elt))
        return names
    if isinstance(expr, ast.Name):
        value = pool.get(expr.id)
        if isinstance(value, list):
            return [str(item) for item in value]
        return [expr.id]
    return []


_LLM_CLASS_PROVIDERS: dict[str, str] = {
    "ChatAnthropic": "anthropic",
    "AsyncAnthropic": "anthropic",
    "ChatOpenAI": "openai",
    "OpenAI": "openai",
    "AzureChatOpenAI": "azure-openai",
    "ChatVertexAI": "google",
    "ChatGoogleGenerativeAI": "google",
    "ChatGroq": "groq",
    "Groq": "groq",
    "ChatMistralAI": "mistral",
    "ChatBedrock": "bedrock",
    "Bedrock": "bedrock",
    "ChatFireworks": "fireworks",
    "Fireworks": "fireworks",
    "ChatCohere": "cohere",
    "ChatAI21": "ai21",
    "ChatXAI": "xai",
    "XAI": "xai",
}


def _resolve_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _dedupe_llm_candidates(candidates: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[dict[str, object]] = []
    for entry in candidates:
        provider = entry.get("provider")
        model = entry.get("model")
        key = (
            provider.lower() if isinstance(provider, str) else None,
            model if isinstance(model, str) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _extract_llm_candidates_from_tree(tree: ast.AST, pool: dict[str, object]) -> list[dict[str, object]]:
    discovered: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _resolve_call_name(node.func)
        if not call_name:
            continue
        short_name = call_name.split(".")[-1]
        provider = _LLM_CLASS_PROVIDERS.get(short_name)
        if not provider:
            continue
        entry: dict[str, object] = {"provider": provider, "source": call_name}
        for kw in node.keywords or []:
            if kw.arg is None:
                continue
            key = kw.arg.lower()
            value = _evaluate_literal(kw.value, pool)
            if key in {"model", "model_name", "model_id"} and isinstance(value, str):
                entry["model"] = value
            elif key == "api_key_env" and isinstance(value, str):
                entry["api_key_env"] = value
            elif key == "temperature" and isinstance(value, (int, float)):
                entry["temperature"] = float(value)
            elif key in {"max_tokens", "max_output_tokens"} and isinstance(value, (int, float)):
                entry["max_output_tokens"] = int(value)
        discovered.append(entry)
    return _dedupe_llm_candidates(discovered)


def _resolve_module_path(project_root: Path, module_name: str | None) -> Path | None:
    if not module_name:
        return None
    parts = module_name.split(".")
    search_roots = [project_root / "src", project_root]
    for root in search_roots:
        candidate = root.joinpath(*parts).with_suffix(".py")
        if candidate.exists():
            return candidate
        package_init = root.joinpath(*parts, "__init__.py")
        if package_init.exists():
            return package_init
    return None


def _build_import_alias_map(tree: ast.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                target = alias.name
                alias_name = alias.asname or alias.name
                if module:
                    mapping[alias_name] = f"{module}.{target}"
                else:
                    mapping[alias_name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                alias_name = alias.asname or alias.name
                mapping[alias_name] = target
    return mapping


def _resolve_call_reference(call_name: str | None, import_map: dict[str, str]) -> str | None:
    if not call_name:
        return None
    if "." in call_name:
        head, *rest = call_name.split(".")
        if head in import_map:
            return ".".join([import_map[head], *rest])
        return call_name
    return import_map.get(call_name, call_name)


def _collect_llm_candidates_recursive(
    project_root: Path,
    module_name: str,
    attr_name: str | None,
    visited: set[str],
) -> list[dict[str, object]]:
    if module_name in visited:
        return []
    visited.add(module_name)
    module_path = _resolve_module_path(project_root, module_name)
    if module_path is None or not module_path.exists():
        return []
    try:
        module_source = module_path.read_text(encoding="utf-8")
        module_tree = ast.parse(module_source, filename=str(module_path))
    except Exception:
        return []
    pool = _build_constant_pool(module_tree)
    candidates = _extract_llm_candidates_from_tree(module_tree, pool)
    if attr_name:
        alias_map = _build_import_alias_map(module_tree)
        for node in module_tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    alias_name = alias.asname or alias.name
                    if alias_name == attr_name and node.module:
                        candidates.extend(
                            _collect_llm_candidates_recursive(
                                project_root,
                                node.module,
                                alias.name,
                                visited,
                            )
                        )
            elif isinstance(node, ast.Assign):
                targets = getattr(node, "targets", [])
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        value = node.value
                        if isinstance(value, ast.Name):
                            resolved = alias_map.get(value.id)
                            if resolved:
                                resolved_module, _, resolved_attr = resolved.rpartition(".")
                                if resolved_module:
                                    candidates.extend(
                                        _collect_llm_candidates_recursive(
                                            project_root,
                                            resolved_module,
                                            resolved_attr or None,
                                            visited,
                                        )
                                    )
                        elif isinstance(value, ast.Attribute):
                            resolved = ast.unparse(value)
                            resolved_module, _, resolved_attr = resolved.rpartition(".")
                            if resolved_module:
                                candidates.extend(
                                    _collect_llm_candidates_recursive(
                                        project_root,
                                        resolved_module,
                                        resolved_attr or None,
                                        visited,
                                    )
                                )
    return _dedupe_llm_candidates(candidates)


def _enrich_metadata_from_assignment_assignment(
    metadata: dict[str, object],
    assignment: ast.AST,
    constant_pool: dict[str, object],
    project_root: Path,
    import_map: dict[str, str],
) -> None:
    if isinstance(assignment, ast.Assign):
        value = assignment.value
    elif isinstance(assignment, ast.AnnAssign):
        value = assignment.value
    else:
        return
    if not isinstance(value, ast.Call):
        return
    raw_factory_call = _format_default(value.func)
    metadata["factory_call"] = raw_factory_call
    resolved_factory = _resolve_call_reference(raw_factory_call, import_map)
    if resolved_factory:
        metadata["factory_resolved"] = resolved_factory
        parts = resolved_factory.split(".")
        if len(parts) >= 2:
            metadata["factory_module"] = ".".join(parts[:-1])
            metadata["factory_attr"] = parts[-1]
    kwargs_map = metadata.setdefault("factory_kwargs", {}) if isinstance(metadata.get("factory_kwargs"), dict) else {}
    metadata["factory_kwargs"] = kwargs_map
    for kw in value.keywords:
        if kw.arg is None:
            continue
        kwargs_map[kw.arg] = _format_default(kw.value)
        if kw.arg == "tools":
            tool_names = _extract_tool_names(kw.value, constant_pool)
            if tool_names:
                metadata["tools"] = [{"name": name} for name in tool_names]
        elif kw.arg in {"system_prompt", "prompt", "instruction", "instructions"}:
            prompt_value = _evaluate_literal(kw.value, constant_pool)
            if isinstance(prompt_value, str):
                metadata.setdefault("prompts", []).append(prompt_value.strip())
        elif kw.arg in {"model", "llm", "model_name"}:
            model_value = _evaluate_literal(kw.value, constant_pool)
            if isinstance(model_value, str):
                overrides = metadata.setdefault("llm_overrides", {})
                overrides[kw.arg] = model_value
    # Probe referenced factory module for implicit LLM defaults
    factory_module = metadata.get("factory_module")
    if isinstance(factory_module, str):
        candidates = _collect_llm_candidates_recursive(
            project_root,
            factory_module,
            metadata.get("factory_attr") if isinstance(metadata.get("factory_attr"), str) else None,
            visited=set(),
        )
        if candidates:
            existing = metadata.setdefault("llm_candidates", [])
            if isinstance(existing, list):
                existing.extend(candidates)
            else:
                metadata["llm_candidates"] = list(candidates)
def _build_parameter_list(args: ast.arguments, *, skip_first: bool) -> list[dict[str, object]]:
    parameters: list[dict[str, object]] = []
    positional = args.args[1:] if skip_first and args.args else args.args or []
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
    if args.kwonlyargs:
        for kw_index, kw_arg in enumerate(args.kwonlyargs):
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


def _extract_parameters(class_node: ast.ClassDef) -> list[dict[str, object]]:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return _build_parameter_list(node.args, skip_first=True)
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
    constant_pool = _build_constant_pool(tree)
    import_map = _build_import_alias_map(tree)
    config_data: list[dict[str, object]] = []

    metadata: dict[str, object] = {
        "module": candidate.module,
        "qualname": candidate.qualname,
        "prompts": list(candidate.signals.get("prompt_literals") or []),
        "tools": _normalise_tools(candidate.signals.get("tool_literals") or []),
        "config_literals": candidate.signals.get("config_literals") or [],
        "factory_functions": candidate.signals.get("factory_functions") or [],
        "instantiations": candidate.signals.get("instantiations"),
        "import_hits": candidate.signals.get("import_hits"),
        "call_targets": candidate.signals.get("call_targets"),
        "framework_hints": candidate.signals.get("framework_hints"),
    }
    class_node = _find_class_definition(tree, candidate.qualname)
    function_node = _find_function_definition(tree, candidate.qualname) if class_node is None else None
    assignment_node = None
    if candidate.factory_kind == "attribute":
        metadata["factory_kind"] = "attribute"
        assignment_node = _find_assignment_definition(tree, candidate.qualname)
    if class_node is not None:
        metadata["parameters"] = _extract_parameters(class_node)
        docstring = ast.get_docstring(class_node)
    elif function_node is not None:
        metadata["parameters"] = _build_parameter_list(function_node.args, skip_first=False)
        docstring = ast.get_docstring(function_node)
    else:
        metadata["parameters"] = []
        docstring = None
    if docstring:
        metadata.setdefault("prompts", []).append(docstring.strip())
    string_literals: list[str] = []
    config_files: list[str] = []
    for node in ast.walk(class_node or function_node or tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if not text:
                continue
            if text.endswith((".json", ".yaml", ".yml")) and "/" in text or text.endswith((".json", ".yaml", ".yml")):
                config_files.append(text)
            elif len(text) > 40 and "You are" in text:
                string_literals.append(text)
    if string_literals:
        metadata.setdefault("prompts", []).extend(string_literals)
    if config_files:
        metadata["config_files"] = config_files
        for rel_path in config_files:
            candidate_path = (candidate.file_path.parent / rel_path).resolve()
            if not candidate_path.exists():
                candidate_path = (project_root / rel_path).resolve()
            if not candidate_path.exists():
                continue
            try:
                if candidate_path.suffix in {".yaml", ".yml"} and yaml is not None:
                    loaded = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
                elif candidate_path.suffix == ".json":
                    loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
                else:
                    continue
            except Exception:
                continue
            if isinstance(loaded, dict):
                config_data.append({"path": str(candidate_path), "content": loaded})
    if config_data:
        metadata["config_data"] = config_data
    if assignment_node is not None:
        _enrich_metadata_from_assignment_assignment(
            metadata,
            assignment_node,
            constant_pool,
            project_root,
            import_map,
        )
    llm_candidates = _extract_llm_candidates_from_tree(tree, constant_pool)
    if llm_candidates:
        existing = metadata.setdefault("llm_candidates", [])
        if isinstance(existing, list):
            existing.extend(llm_candidates)
        else:
            metadata["llm_candidates"] = list(llm_candidates)
    return metadata
