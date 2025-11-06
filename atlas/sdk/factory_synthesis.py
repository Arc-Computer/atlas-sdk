"""LLM-assisted factory synthesis for Atlas autodiscovery."""

from __future__ import annotations

import ast
import json
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from atlas.cli.utils import CLIError
from atlas.config.models import LLMParameters, LLMProvider
from atlas.sdk.discovery import Candidate
from atlas.utils.llm_client import LLMClient

try:  # pragma: no cover - tomllib is unavailable on older interpreters
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

RoleLiteral = str  # alias for readability

ENV_FUNCTION_NAME = "create_environment"
AGENT_FUNCTION_NAME = "create_agent"
GENERATED_MODULE = "atlas_generated_factories"
ENV_VALIDATE_FLAG = "ATLAS_DISCOVERY_VALIDATE"

# Runtime helper code injected into generated factory files
AGENT_RUNTIME_HELPER = textwrap.dedent(
    """\
    _AGENT_RUNTIME_CACHE = {}
    _VALIDATION_MARKER = ".validated"
    try:
        from langchain_core.messages import BaseMessage  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        BaseMessage = None  # type: ignore
    try:
        from langchain_core.messages import HumanMessage  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        HumanMessage = None  # type: ignore
    try:
        from collections.abc import Mapping
    except Exception:  # pragma: no cover - fallback
        Mapping = dict  # type: ignore

    def _atlas_require_validation():
        marker = os.path.join(os.path.dirname(__file__), _VALIDATION_MARKER)
        if os.path.exists(marker) or os.environ.get("ATLAS_SKIP_VALIDATION") == "1":
            return
        raise RuntimeError(
            "Atlas generated factories have not been validated. "
            "Run 'atlas env init' to regenerate and validate factories, "
            "or set ATLAS_SKIP_VALIDATION=1 to bypass this check."
        )

    def _atlas_jsonable(value, depth=0):
        if depth > 6:
            return str(value)
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.decode("utf-8", "ignore")
        if BaseMessage is not None and isinstance(value, BaseMessage):
            payload = {"type": getattr(value, "type", None) or value.__class__.__name__.lower()}
            payload["content"] = _atlas_jsonable(getattr(value, "content", None), depth + 1)
            tool_calls = getattr(value, "tool_calls", None)
            if tool_calls:
                payload["tool_calls"] = [_atlas_jsonable(call, depth + 1) for call in tool_calls]
            additional_kwargs = getattr(value, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict) and additional_kwargs:
                payload["additional_kwargs"] = _atlas_jsonable(additional_kwargs, depth + 1)
            return payload
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                if isinstance(dumped, dict):
                    return _atlas_jsonable(dumped, depth + 1)
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                dumped = value.dict()
                if isinstance(dumped, dict):
                    return _atlas_jsonable(dumped, depth + 1)
            except Exception:
                pass
        if isinstance(value, Mapping):
            normalised = {}
            for key, item in value.items():
                normalised[str(key)] = _atlas_jsonable(item, depth + 1)
            return normalised
        if isinstance(value, (list, tuple, set)):
            return [_atlas_jsonable(item, depth + 1) for item in value]
        if hasattr(value, "__dict__"):
            return _atlas_jsonable(vars(value), depth + 1)
        return str(value)

    def _atlas_normalise_agent_output(result):
        processed = _atlas_jsonable(result)
        if isinstance(processed, str):
            return processed
        if isinstance(processed, (int, float, bool)) or processed is None:
            return str(processed)
        try:
            return json.dumps(processed, ensure_ascii=False)
        except Exception:
            return str(processed)

    def _atlas_execute_agent(agent, prompt, metadata=None):
        call_kwargs = {}
        payload_hints = []
        if isinstance(metadata, dict):
            candidate = metadata.get("payload")
            if candidate is not None:
                payload_hints.append(candidate)
            extra = metadata.get("call_kwargs")
            if isinstance(extra, dict):
                call_kwargs = dict(extra)
        attempts = list(payload_hints)
        attempts.append(prompt)
        attempts.append({"input": prompt})
        if HumanMessage is not None:
            attempts.append({"messages": [HumanMessage(content=prompt)]})
        else:
            attempts.append({"messages": [{"role": "user", "content": prompt}]})
        last_error = None
        if hasattr(agent, "invoke"):
            for payload in attempts:
                try:
                    if call_kwargs:
                        response = agent.invoke(payload, **call_kwargs)
                    else:
                        response = agent.invoke(payload)
                    return _atlas_normalise_agent_output(response)
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:
                    last_error = exc
                    continue
        if callable(agent):
            try:
                return _atlas_normalise_agent_output(agent(prompt))
            except TypeError:
                try:
                    return _atlas_normalise_agent_output(agent(prompt, metadata=metadata))
                except TypeError as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
        if hasattr(agent, "run"):
            try:
                return _atlas_normalise_agent_output(agent.run(prompt))
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Atlas runtime adapter could not execute agent") from last_error
    """
)
_ENTRYPOINT_KEYWORDS = (
    "create",
    "bootstrap",
    "start",
    "launch",
    "init",
    "environment",
    "agent",
    "session",
)
_SKIP_REPO_DIRS = {
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
    ".atlas",
}


@dataclass(slots=True)
class ClassContext:
    """Static insights about a candidate class."""

    module: str
    qualname: str
    file_path: Path
    docstring: str | None
    class_source: str
    init_signature: str
    required_args: list[str]
    optional_args: list[str]
    provided_kwargs: dict[str, Any]
    constants: list[str]
    sample_invocations: list[str]


@dataclass(slots=True)
class FactorySnippet:
    """Structured response from the LLM for a single factory.

    Attributes:
        function_name: Name of the generated factory function
        imports: List of import statements required
        helpers: Helper functions to include in generated file
        factory_body: Complete function definition source code
        internal_notes: Developer-facing notes about implementation
        prerequisites: User-actionable prerequisites (packages, env vars, etc.)
        preflight: Legacy field for preflight checks
        auto_skip: Whether to skip validation for this factory
        notes: Deprecated - migrated to internal_notes on init
    """

    function_name: str
    imports: list[str] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)
    factory_body: str = ""
    internal_notes: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    preflight: list[str] = field(default_factory=list)
    auto_skip: bool = False

    # Deprecated - maintained for backward compatibility
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Migrate legacy 'notes' field to 'internal_notes' if present."""
        if self.notes and not self.internal_notes and not self.prerequisites:
            # Legacy support: treat all old notes as internal notes
            self.internal_notes = list(self.notes)


@dataclass(slots=True)
class SynthesisOutcome:
    """Aggregated synthesis result for environment/agent."""

    environment_factory: tuple[str, str] | None = None
    agent_factory: tuple[str, str] | None = None
    preflight_notes: list[str] = field(default_factory=list)
    auxiliary_notes: list[str] = field(default_factory=list)
    auto_skip: bool = False
    environment_auto_wrapped: bool = False
    agent_auto_wrapped: bool = False


@dataclass(slots=True)
class RepositorySummary:
    """Lightweight summary of project structure for wrapper synthesis."""

    role: RoleLiteral
    project_name: str
    readme_excerpt: str
    dependencies: list[str]
    cli_scripts: list[str]
    entrypoints: list[dict[str, str]]
    framework_hints: list[str]
    notable_calls: list[str]
    provided_kwargs: dict[str, Any]


class FactorySynthesizer:
    """Coordinates LLM calls and module emission for generated factories."""

    def __init__(
        self,
        project_root: Path,
        atlas_dir: Path,
        *,
        llm_models: Sequence[LLMParameters] | None = None,
    ) -> None:
        self._project_root = project_root
        self._atlas_dir = atlas_dir
        self._atlas_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_init()
        self._llm_models = list(llm_models or self._default_models())
        self._clients: list[LLMClient] | None = None
        self._context_cache: dict[RoleLiteral, ClassContext | RepositorySummary] = {}
        self._snippet_cache: dict[RoleLiteral, FactorySnippet] = {}
        self._error_history: dict[RoleLiteral, list[str]] = {}
        self._repository_cache: dict[str, Any] | None = None
        self._analysis_cache: dict[RoleLiteral, tuple[bool, ClassContext, list[str]]] = {}

    def synthesise(
        self,
        *,
        environment: Candidate | None,
        agent: Candidate | None,
        environment_kwargs: dict[str, Any],
        agent_kwargs: dict[str, Any],
        environment_summary: RepositorySummary | None = None,
        agent_summary: RepositorySummary | None = None,
    ) -> SynthesisOutcome:
        outcome = SynthesisOutcome()
        snippets: dict[str, FactorySnippet] = {}

        if environment is not None or environment_summary is not None:
            env_context: ClassContext | RepositorySummary
            if environment is not None:
                env_needed, env_context, _env_missing = self._get_candidate_analysis(environment, environment_kwargs)
            else:
                env_context = environment_summary or self._build_repository_summary(
                    "environment",
                    provided_kwargs=environment_kwargs,
                )
                env_needed = True
            if env_needed:
                snippet = self._generate_snippet("environment", env_context, previous_error=None)
                snippets["environment"] = snippet
                self._snippet_cache["environment"] = snippet
                outcome.environment_factory = (GENERATED_MODULE, snippet.function_name)
                outcome.preflight_notes.extend(snippet.preflight)
                outcome.auxiliary_notes.extend(snippet.notes)
                outcome.auto_skip = outcome.auto_skip or snippet.auto_skip
                outcome.environment_auto_wrapped = isinstance(env_context, RepositorySummary)

        if agent is not None or agent_summary is not None:
            agent_context: ClassContext | RepositorySummary
            if agent is not None:
                agent_needed, agent_context, _agent_missing = self._get_candidate_analysis(agent, agent_kwargs)
            else:
                agent_context = agent_summary or self._build_repository_summary(
                    "agent",
                    provided_kwargs=agent_kwargs,
                )
                agent_needed = True
            if agent_needed:
                snippet = self._generate_snippet("agent", agent_context, previous_error=None)
                snippets["agent"] = snippet
                self._snippet_cache["agent"] = snippet
                outcome.agent_factory = (GENERATED_MODULE, snippet.function_name)
                outcome.preflight_notes.extend(snippet.preflight)
                outcome.auxiliary_notes.extend(snippet.notes)
                outcome.auto_skip = outcome.auto_skip or snippet.auto_skip
                outcome.agent_auto_wrapped = isinstance(agent_context, RepositorySummary)

        if snippets:
            self._emit_module(dict(self._snippet_cache))

        return outcome

    def retry_with_error(self, error_text: str) -> None:
        """Regenerate factories after a worker error."""
        snippets: dict[str, FactorySnippet] = {}
        for role, snippet in self._snippet_cache.items():
            context = self._context_cache.get(role)
            if context is None:
                continue
            history = self._error_history.setdefault(role, [])
            history.append(error_text)
            new_snippet = self._generate_snippet(role, context, previous_error="\n\n".join(history))
            snippets[role] = new_snippet
            self._snippet_cache[role] = new_snippet
        if snippets:
            self._emit_module(dict(self._snippet_cache))

    def prepare_repository_summary(
        self,
        role: RoleLiteral,
        *,
        provided_kwargs: dict[str, Any],
    ) -> RepositorySummary:
        """Expose repository summary for callers that need to inspect it."""
        return self._build_repository_summary(role, provided_kwargs=provided_kwargs)

    def emit_manual_snippets(self, snippets: dict[str, FactorySnippet]) -> None:
        """Write caller-provided snippets without invoking LLM synthesis."""

        if not snippets:
            return
        for role, snippet in snippets.items():
            self._snippet_cache[role] = snippet
        self._emit_module(dict(self._snippet_cache))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_package_init(self) -> None:
        init_path = self._atlas_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                "# Auto-generated package marker for Atlas discovery artefacts.\n",
                encoding="utf-8",
            )

    def _get_candidate_analysis(
        self,
        candidate: Candidate,
        provided_kwargs: dict[str, Any],
    ) -> tuple[bool, ClassContext, list[str]]:
        cached = self._analysis_cache.get(candidate.role)
        if cached is not None:
            return cached
        result = self._analyse_candidate(candidate, provided_kwargs)
        self._analysis_cache[candidate.role] = result
        self._context_cache[candidate.role] = result[1]
        return result

    def needs_factory_for_candidate(
        self,
        candidate: Candidate,
        provided_kwargs: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        needs_factory, _context, missing = self._get_candidate_analysis(candidate, provided_kwargs)
        return needs_factory, missing

    def _analyse_candidate(
        self,
        candidate: Candidate,
        provided_kwargs: dict[str, Any],
    ) -> tuple[bool, ClassContext, list[str]]:
        file_path = candidate.file_path
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        class_node = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == candidate.qualname),
            None,
        )
        if class_node is None:
            raise CLIError(f"Unable to locate class definition for {candidate.dotted_path()}")
        docstring = ast.get_docstring(class_node)
        class_source = textwrap.dedent(ast.get_source_segment(source, class_node) or "").strip()
        init_signature, required_args, optional_args = self._extract_init_signature(class_node)
        constants = self._extract_constants(tree, source)
        usage = self._collect_sample_usage(candidate.qualname)
        context = ClassContext(
            module=candidate.module,
            qualname=candidate.qualname,
            file_path=file_path,
            docstring=docstring,
            class_source=self._truncate(class_source, 1200),
            init_signature=init_signature,
            required_args=required_args,
            optional_args=optional_args,
            provided_kwargs=provided_kwargs,
            constants=constants,
            sample_invocations=usage,
        )
        missing = [arg for arg in required_args if arg not in provided_kwargs]
        return (len(missing) > 0), context, missing

    def _collect_sample_usage(self, class_name: str) -> list[str]:
        try:
            command = [
                "rg",
                "--no-heading",
                "--color",
                "never",
                "--max-count",
                "5",
                f"{class_name}\\(",
            ]
            result = subprocess.run(
                command,
                cwd=self._project_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception:
            return []
        if result.returncode not in {0, 1}:
            return []
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[:3]

    def _emit_module(self, snippets: dict[str, FactorySnippet]) -> None:
        header = [
            "# This module was generated by `atlas env init` to provide autodiscovery factories.",
            "# Edits will be overwritten on subsequent discovery runs.",
            "from __future__ import annotations",
            "",
        ]
        imports: list[str] = []
        helpers: list[str] = []
        functions: list[str] = []
        for snippet in snippets.values():
            imports.extend(snippet.imports)
            helpers.extend(snippet.helpers)
            functions.append(snippet.factory_body.rstrip())
        unique_imports = self._deduplicate_preserve_order(imports)
        content_lines = header + unique_imports
        if unique_imports and unique_imports[-1] != "":
            content_lines.append("")
        if helpers:
            content_lines.extend([helper.rstrip() + "\n" for helper in helpers])
        if functions:
            if helpers:
                content_lines.append("")
            content_lines.extend(functions)
            if not functions[-1].endswith("\n"):
                content_lines.append("")
        target = self._atlas_dir / "generated_factories.py"
        content = "\n".join(line.rstrip() for line in content_lines).rstrip() + "\n"
        target.write_text(content, encoding="utf-8")
        self._ensure_alias_module()

    def _ensure_alias_module(self) -> None:
        alias_path = self._project_root / f"{GENERATED_MODULE}.py"
        alias_content = textwrap.dedent(
            """
            # Auto-generated by `atlas env init`. Mirrors .atlas/generated_factories for runtime imports.
            import importlib.util as _atlas_util
            import pathlib as _atlas_pathlib
            import sys as _atlas_sys

            _atlas_target = _atlas_pathlib.Path(__file__).resolve().parent / '.atlas' / 'generated_factories.py'
            _atlas_spec = _atlas_util.spec_from_file_location('_atlas_env_generated_factories', _atlas_target)
            _atlas_module = _atlas_util.module_from_spec(_atlas_spec)
            _atlas_spec.loader.exec_module(_atlas_module)
            _atlas_sys.modules.setdefault('atlas_generated_factories', _atlas_module)
            _atlas_sys.modules.setdefault('.atlas.generated_factories', _atlas_module)
            globals().update({k: getattr(_atlas_module, k) for k in dir(_atlas_module) if not k.startswith('_')})
            """
        ).lstrip()
        alias_path.write_text(alias_content, encoding="utf-8")

    def _generate_snippet(
        self,
        role: RoleLiteral,
        context: ClassContext | RepositorySummary,
        *,
        previous_error: str | None,
    ) -> FactorySnippet:
        if isinstance(context, ClassContext):
            prompt_payload = {
                "role": role,
                "class_module": context.module,
                "class_name": context.qualname,
                "docstring": context.docstring or "",
                "init_signature": context.init_signature,
                "required_args": context.required_args,
                "optional_args": context.optional_args,
                "provided_kwargs": context.provided_kwargs,
                "class_source": context.class_source,
                "constants": context.constants,
                "sample_invocations": context.sample_invocations,
                "validate_env_flag": ENV_VALIDATE_FLAG,
                "previous_error": previous_error or "",
            }
        else:
            prompt_payload = {
                "role": role,
                "class_module": "",
                "class_name": "",
                "docstring": "",
                "init_signature": "",
                "required_args": [],
                "optional_args": [],
                "provided_kwargs": context.provided_kwargs,
                "class_source": "",
                "constants": [],
                "sample_invocations": [],
                "validate_env_flag": ENV_VALIDATE_FLAG,
                "previous_error": previous_error or "",
                "repository_summary": {
                    "project_name": context.project_name,
                    "readme_excerpt": context.readme_excerpt,
                    "dependencies": context.dependencies,
                    "cli_scripts": context.cli_scripts,
                    "entrypoints": context.entrypoints,
                    "framework_hints": context.framework_hints,
                    "notable_calls": context.notable_calls,
                    "provided_kwargs": context.provided_kwargs,
                    "role": context.role,
                },
            }
        messages = [
            {
                "role": "system",
                "content": _SYNTHESIS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False),
            },
        ]
        clients = self._ensure_clients()
        last_error: Exception | None = None
        for client in clients:
            try:
                response = client.complete(
                    messages,
                    overrides={"temperature": 0.2, "max_tokens": 8192},
                )
            except Exception as exc:  # pragma: no cover - network/backoff handled upstream
                last_error = exc
                continue
            snippet = self._parse_response(response.content, role=role)
            if snippet is not None:
                return snippet
        raise CLIError(
            f"LLM synthesis failed for {role} factory."
            + (f" Last error: {last_error}" if last_error else "")
        )

    def _parse_response(self, raw: str, *, role: RoleLiteral) -> FactorySnippet | None:
        candidate_text = raw.strip()
        if candidate_text.startswith("```"):
            candidate_text = self._extract_code_block(candidate_text)
        try:
            payload = json.loads(candidate_text)
        except json.JSONDecodeError:
            return None
        function_name = payload.get("function_name") or (
            ENV_FUNCTION_NAME if role == "environment" else AGENT_FUNCTION_NAME
        )
        imports = self._normalize_string_list(payload.get("imports"))
        helpers = self._normalize_string_list(payload.get("helpers"))
        factory_body = payload.get("factory") or ""
        notes = self._normalize_string_list(payload.get("notes"))
        preflight = self._normalize_string_list(payload.get("preflight"))
        auto_skip = bool(payload.get("auto_skip", False))
        decoded_helpers = [self._unescape(helper) for helper in helpers]
        decoded_factory = self._unescape(factory_body)
        snippet = FactorySnippet(
            function_name=function_name,
            imports=imports,
            helpers=decoded_helpers,
            factory_body=decoded_factory,
            notes=notes,
            preflight=preflight,
            auto_skip=auto_skip,
        )
        return snippet

    def _ensure_clients(self) -> list[LLMClient]:
        if self._clients is not None:
            return self._clients
        clients: list[LLMClient] = []
        last_error: Exception | None = None
        for params in self._llm_models:
            try:
                clients.append(LLMClient(params))
            except Exception as exc:  # pragma: no cover - missing dependencies
                last_error = exc
                continue
        if not clients:
            raise CLIError(
                "LLM synthesis requires a configured provider. "
                "Set GEMINI_API_KEY or ANTHROPIC_API_KEY before running discovery."
                + (f" Last error: {last_error}" if last_error else "")
            )
        self._clients = clients
        return clients

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value if item is not None]
        return []

    @staticmethod
    def _unescape(payload: str) -> str:
        return payload.encode("utf-8").decode("unicode_escape")

    @staticmethod
    def _extract_code_block(payload: str) -> str:
        stripped = payload.strip().strip("`")
        for prefix in ("python", "json"):
            if stripped.lstrip().startswith(prefix):
                stripped = stripped.lstrip()[len(prefix):]
                break
        return stripped.strip()

    @staticmethod
    def _deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        if result and result[-1].strip():
            result.append("")
        return result

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 20] + "\n# ... truncated ..."

    def _extract_constants(self, tree: ast.AST, source: str) -> list[str]:
        snippets: list[str] = []
        for node in tree.body:  # type: ignore[attr-defined]
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    segment = ast.get_source_segment(source, node) or ""
                    cleaned = textwrap.dedent(segment).strip()
                    snippets.append(self._truncate(cleaned, 400))
        return snippets[:3]

    @staticmethod
    def _extract_init_signature(node: ast.ClassDef) -> tuple[str, list[str], list[str]]:
        init_func = next(
            (child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"),
            None,
        )
        if init_func is None:
            return "def __init__(self) -> None", [], []
        args = init_func.args
        arg_names = [arg.arg for arg in args.args][1:]  # skip self
        defaults = args.defaults or []
        required_count = len(arg_names) - len(defaults)
        required_args = arg_names[:required_count]
        optional_args = arg_names[required_count:]
        kwonly_required = [
            arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults or []) if default is None
        ]
        required_args.extend(kwonly_required)

        signature = f"def __init__{ast.unparse(init_func.args)}"
        return signature, required_args, optional_args

    def _collect_repository_data(self) -> dict[str, Any]:
        if self._repository_cache is not None:
            return self._repository_cache
        readme_excerpt = self._read_readme_excerpt()
        project_name = self._detect_project_name()
        dependencies = self._collect_dependencies()
        cli_scripts = self._collect_cli_scripts()
        entrypoints, notable_calls = self._collect_entrypoint_hints()
        framework_hints = self._detect_framework_hints(dependencies, notable_calls, readme_excerpt)
        self._repository_cache = {
            "project_name": project_name,
            "readme_excerpt": readme_excerpt,
            "dependencies": dependencies,
            "cli_scripts": cli_scripts,
            "entrypoints": entrypoints,
            "framework_hints": framework_hints,
            "notable_calls": notable_calls,
        }
        return self._repository_cache

    def _build_repository_summary(
        self,
        role: RoleLiteral,
        *,
        provided_kwargs: dict[str, Any],
    ) -> RepositorySummary:
        base = self._collect_repository_data()
        summary = RepositorySummary(
            role=role,
            project_name=base["project_name"],
            readme_excerpt=base["readme_excerpt"],
            dependencies=base["dependencies"],
            cli_scripts=base["cli_scripts"],
            entrypoints=base["entrypoints"],
            framework_hints=base["framework_hints"],
            notable_calls=base["notable_calls"],
            provided_kwargs=provided_kwargs,
        )
        self._context_cache[role] = summary
        return summary

    def _read_readme_excerpt(self) -> str:
        for name in ("README.md", "readme.md", "README.rst", "README"):
            candidate = self._project_root / name
            if candidate.exists():
                try:
                    text = candidate.read_text(encoding="utf-8")
                except Exception:
                    continue
                return self._truncate(textwrap.dedent(text.strip()), 1600)
        return ""

    def _detect_project_name(self) -> str:
        if tomllib is None:
            return self._project_root.name
        pyproject = self._project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except Exception:
                return self._project_root.name
            project = data.get("project") or {}
            if isinstance(project, dict):
                name = project.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
            tool = data.get("tool") or {}
            if isinstance(tool, dict):
                poetry = tool.get("poetry") or {}
                if isinstance(poetry, dict):
                    name = poetry.get("name")
                    if isinstance(name, str) and name.strip():
                        return name.strip()
        return self._project_root.name

    def _collect_dependencies(self) -> list[str]:
        dependencies: list[str] = []
        if tomllib is not None:
            pyproject = self._project_root / "pyproject.toml"
            if pyproject.exists():
                try:
                    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                dependencies.extend(self._extract_dependencies_from_pyproject(payload))
        requirements = self._project_root / "requirements.txt"
        if requirements.exists():
            try:
                for line in requirements.read_text(encoding="utf-8").splitlines():
                    cleaned = line.strip()
                    if not cleaned or cleaned.startswith("#"):
                        continue
                    dependencies.append(cleaned.split(";")[0].strip())
            except Exception:
                pass
        unique = []
        seen: set[str] = set()
        for dep in dependencies:
            normalized = dep.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique[:20]

    @staticmethod
    def _extract_dependencies_from_pyproject(payload: dict[str, Any]) -> list[str]:
        deps: list[str] = []
        project = payload.get("project")
        if isinstance(project, dict):
            raw = project.get("dependencies") or []
            if isinstance(raw, list):
                deps.extend(str(item) for item in raw)
        tool = payload.get("tool")
        if isinstance(tool, dict):
            poetry = tool.get("poetry")
            if isinstance(poetry, dict):
                poetry_deps = poetry.get("dependencies")
                if isinstance(poetry_deps, dict):
                    deps.extend(poetry_deps.keys())
        return deps

    def _collect_cli_scripts(self) -> list[str]:
        if tomllib is None:
            return []
        pyproject = self._project_root / "pyproject.toml"
        if not pyproject.exists():
            return []
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception:
            return []
        scripts: list[str] = []
        project = payload.get("project")
        if isinstance(project, dict):
            entry_points = project.get("scripts")
            if isinstance(entry_points, dict):
                scripts.extend(f"{name}={value}" for name, value in entry_points.items())
        tool = payload.get("tool")
        if isinstance(tool, dict):
            poetry = tool.get("poetry")
            if isinstance(poetry, dict):
                ep = poetry.get("scripts")
                if isinstance(ep, dict):
                    scripts.extend(f"{name}={value}" for name, value in ep.items())
        return scripts[:20]

    def _collect_entrypoint_hints(self) -> tuple[list[dict[str, str]], list[str]]:
        entrypoints: list[dict[str, str]] = []
        notable_calls: list[str] = []
        limit = 24
        for path in sorted(self._project_root.rglob("*.py")):
            if self._should_skip_path(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            module_name = self._module_name_from_path(path)
            if not module_name:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and self._looks_like_entrypoint(node.name):
                    docstring = ast.get_docstring(node) or ""
                    signature = f"def {node.name}{ast.unparse(node.args)}"
                    snippet = ast.get_source_segment(source, node) or ""
                    entrypoints.append(
                        {
                            "module": module_name,
                            "symbol": node.name,
                            "kind": "function",
                            "signature": self._truncate(textwrap.dedent(signature).strip(), 200),
                            "docstring": self._truncate(docstring.strip(), 240),
                            "source_excerpt": self._truncate(textwrap.dedent(snippet).strip(), 400),
                        }
                    )
                if isinstance(node, ast.ClassDef) and self._looks_like_entrypoint(node.name):
                    docstring = ast.get_docstring(node) or ""
                    signature = f"class {node.name}"
                    snippet = ast.get_source_segment(source, node) or ""
                    entrypoints.append(
                        {
                            "module": module_name,
                            "symbol": node.name,
                            "kind": "class",
                            "signature": self._truncate(signature, 200),
                            "docstring": self._truncate(docstring.strip(), 240),
                            "source_excerpt": self._truncate(textwrap.dedent(snippet).strip(), 400),
                        }
                    )
            notable_calls.extend(self._collect_function_calls(tree))
            if len(entrypoints) >= limit:
                break
        return entrypoints[:limit], list(dict.fromkeys(notable_calls))[:30]

    def _collect_function_calls(self, tree: ast.AST) -> list[str]:
        hints: list[str] = []
        target_names = {
            "create_deep_agent",
            "create_langgraph_agent",
            "create_agent",
            "create_environment",
            "build_environment",
            "start_session",
            "run_session",
            "Session",
            "Environment",
            "Agent",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    continue
                if name in target_names:
                    hints.append(name)
        return hints

    def _detect_framework_hints(
        self,
        dependencies: list[str],
        notable_calls: list[str],
        readme_excerpt: str,
    ) -> list[str]:
        keywords = {
            "deepagents": "deepagents",
            "langgraph": "langgraph",
            "secrl": "secrl",
            "secgym": "secgym",
            "autogen": "autogen",
            "gymnasium": "gymnasium",
            "openai": "openai",
            "anthropic": "anthropic",
            "tavily": "tavily",
        }
        hints: set[str] = set()
        sources = " ".join(
            [readme_excerpt.lower()]
            + [dep.lower() for dep in dependencies]
            + [call.lower() for call in notable_calls]
        )
        for key, label in keywords.items():
            if key in sources:
                hints.add(label)
        return sorted(hints)

    def _should_skip_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(parts & _SKIP_REPO_DIRS)

    def _module_name_from_path(self, path: Path) -> str:
        rel = path.relative_to(self._project_root)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[0] in {"src", "source", "python"}:
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return ""
        return ".".join(parts)

    @staticmethod
    def _looks_like_entrypoint(name: str) -> bool:
        lowered = name.lower()
        return any(keyword in lowered for keyword in _ENTRYPOINT_KEYWORDS)

    @staticmethod
    def _default_models() -> list[LLMParameters]:
        return [
            LLMParameters(
                provider=LLMProvider.ANTHROPIC,
                model="claude-haiku-4-5",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.1,
                max_output_tokens=16000,
                timeout_seconds=120.0,
            ),
            LLMParameters(
                provider=LLMProvider.GEMINI,
                model="gemini/gemini-2.5-flash",
                api_key_env="GEMINI_API_KEY",
                temperature=0.1,
                max_output_tokens=16000,
                timeout_seconds=120.0,
            ),
        ]


_SYNTHESIS_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are assisting Atlas CLI in generating Python factory helpers for autodiscovery.
    Each factory MUST:
    - Import the target class directly from its module path (or synthesized wrapper).
    - Provide deterministic defaults for required constructor arguments using the context provided.
    - Accept arbitrary keyword overrides and merge them with the defaults.
    - Delay side effects (network, Docker, database) until the environment variable {ENV_VALIDATE_FLAG} is "1".
      When validation is not enabled, raise RuntimeError with a useful message and include any preflight guidance
      in the generated metadata.
    - Avoid importing heavyweight modules unless validation is true (wrap them inside the validate branch).
    - Return a fully constructed instance when validation is enabled.

    When repository_summary data is present (no explicit class definition was discovered):
    - Synthesize lightweight wrapper classes that satisfy the Atlas environment or agent protocol for the given role.
    - Place the wrapper class definition(s) inside the "helpers" array so they are emitted into the module.
    - Use entrypoint hints, dependencies, and framework references to import the correct primitives.
    - Set the "auto_skip" flag to true when prerequisites are missing or validation is required before execution.

    Respond with strict JSON (no markdown fences) containing:
      - "function_name": name of the factory function to emit (snake_case).
      - "imports": array of import statements required at module scope (strings).
      - "helpers": array of helper function/constant/class definitions (each as a single string) or an empty array.
      - "factory": the full function definition implementing the factory.
      - "notes": array of short textual notes about assumptions or TODOs (may be empty).
      - "preflight": array of warnings/preflight steps to surface to the user (may be empty).
      - "auto_skip": boolean indicating if discovery should auto-skip running the loop until validation.

    Ensure the resulting code is valid Python 3.12 and uses typing hints where reasonable.
    """
).strip()


# ============================================================================
# Factory Building Functions
# ============================================================================
# These functions generate FactorySnippet objects for environment and agent
# factories discovered during autodiscovery. Extracted from atlas/cli/env.py
# during refactoring to consolidate factory generation logic.


def format_kwargs_literal(payload: dict[str, object]) -> str:
    """Format a dictionary of keyword arguments as a JSON literal string.

    Args:
        payload: Dictionary to format

    Returns:
        JSON string representation with 4-space indentation, or "{}" if empty
    """
    if not payload:
        return "{}"
    return json.dumps(payload, indent=4, sort_keys=True)


def build_basic_environment_factory_snippet(candidate: Candidate, defaults: dict[str, object]) -> FactorySnippet:
    """Build factory snippet for a basic (non-factory) environment class.

    Generates a factory function that instantiates the environment class with
    optional default kwargs. Includes validation guard to prevent execution
    during discovery.

    Args:
        candidate: Discovered environment candidate
        defaults: Default keyword arguments to merge

    Returns:
        FactorySnippet ready for code generation
    """
    target_symbol = candidate.qualname
    call_symbol = target_symbol
    if target_symbol == ENV_FUNCTION_NAME:
        call_symbol = f"_repo_{ENV_FUNCTION_NAME}"
        import_binding = f"from {candidate.module} import {target_symbol} as {call_symbol}"
    else:
        import_binding = f"from {candidate.module} import {target_symbol}"
    imports = [import_binding]
    defaults_literal = format_kwargs_literal(defaults)
    preamble = ""
    lines: list[str] = []
    if defaults:
        preamble = f"DEFAULT_ENVIRONMENT_KWARGS = {defaults_literal}\n\n"
        lines.append("    parameters = dict(DEFAULT_ENVIRONMENT_KWARGS)")
        lines.append("    parameters.update(kwargs)")
    else:
        lines.append("    parameters = dict(kwargs)")
    lines.append("    _atlas_require_validation()")
    lines.append(f"    return {call_symbol}(**parameters)")
    body_lines = [
        f"def {ENV_FUNCTION_NAME}(**kwargs):",
        '    """Atlas-generated environment factory."""',
        *lines,
    ]
    factory_body = preamble + "\n".join(body_lines)
    internal_notes = [f"Generated environment factory wrapping {candidate.dotted_path()}."]
    return FactorySnippet(
        function_name=ENV_FUNCTION_NAME,
        imports=imports,
        helpers=[],
        factory_body=factory_body,
        internal_notes=internal_notes,
        prerequisites=[],
        preflight=[],
        auto_skip=False,
    )


def build_function_environment_factory_snippet(candidate: Candidate) -> FactorySnippet:
    """Build factory snippet for a factory function/attribute environment.

    Handles both module-level callables and attributes. For attributes, uses
    importlib for dynamic loading.

    Args:
        candidate: Discovered environment factory candidate

    Returns:
        FactorySnippet ready for code generation
    """
    if candidate.factory_kind == "attribute":
        imports = ["import importlib", "import os"]
        body_lines = [
            f"def {ENV_FUNCTION_NAME}(**kwargs):",
            '    """Atlas-generated environment factory wrapping repository attribute."""',
            "    _atlas_require_validation()",
            f"    module = importlib.import_module('{candidate.module}')",
            f"    target = getattr(module, '{candidate.qualname}')",
            "    if callable(target):",
            "        return target(**kwargs)",
            "    if kwargs:",
            "        raise TypeError('Environment factory does not accept keyword arguments.')",
            "    return target",
        ]
    else:
        target_symbol = candidate.qualname
        call_symbol = target_symbol
        if target_symbol == ENV_FUNCTION_NAME:
            call_symbol = f"_repo_{ENV_FUNCTION_NAME}"
            import_binding = f"from {candidate.module} import {target_symbol} as {call_symbol}"
        else:
            import_binding = f"from {candidate.module} import {target_symbol}"
        imports = ["import os", import_binding]
        body_lines = [
            f"def {ENV_FUNCTION_NAME}(**kwargs):",
            '    """Atlas-generated environment factory wrapping repository callable."""',
            "    _atlas_require_validation()",
            f"    return {call_symbol}(**kwargs)",
        ]
    factory_body = "\n".join(body_lines)
    internal_notes = [f"Delegates to {candidate.module}:{candidate.qualname}."]
    return FactorySnippet(
        function_name=ENV_FUNCTION_NAME,
        imports=imports,
        helpers=[],
        factory_body=factory_body,
        internal_notes=internal_notes,
        prerequisites=[],
        preflight=[],
        auto_skip=False,
    )


def build_basic_agent_factory_snippet(candidate: Candidate, defaults: dict[str, object]) -> FactorySnippet:
    """Build factory snippet for a basic (non-factory) agent class.

    Generates a factory function that supports both:
    - Discovery mode: Instantiates and wraps in StepwiseAgentAdapter
    - Runtime mode: Caches instance and executes via _atlas_execute_agent helper

    Args:
        candidate: Discovered agent candidate
        defaults: Default keyword arguments to merge

    Returns:
        FactorySnippet with AGENT_RUNTIME_HELPER included
    """
    target_symbol = candidate.qualname
    call_symbol = target_symbol
    if target_symbol == AGENT_FUNCTION_NAME:
        call_symbol = f"_repo_{AGENT_FUNCTION_NAME}"
        import_binding = f"from {candidate.module} import {target_symbol} as {call_symbol}"
    else:
        import_binding = f"from {candidate.module} import {target_symbol}"
    imports = [
        "import json",
        "import os",
        import_binding,
        "from atlas.sdk.wrappers import StepwiseAgentAdapter",
    ]
    defaults_literal = format_kwargs_literal(defaults)
    preamble = ""
    lines: list[str] = [
        "    has_prompt = 'prompt' in kwargs",
        "    has_metadata = 'metadata' in kwargs",
        "    prompt = kwargs.pop('prompt', None)",
        "    metadata = kwargs.pop('metadata', None)",
        "    runtime_invocation = has_prompt or has_metadata",
    ]
    if defaults:
        preamble = f"DEFAULT_AGENT_KWARGS = {defaults_literal}\n\n"
        lines.append("    parameters = dict(DEFAULT_AGENT_KWARGS)")
        lines.append("    parameters.update(kwargs)")
    else:
        lines.append("    parameters = dict(kwargs)")
    lines.extend(
        [
            f"    cache_key = ('agent', '{candidate.module}', '{candidate.qualname}')",
            "    if runtime_invocation:",
            "        agent_instance = _AGENT_RUNTIME_CACHE.get(cache_key)",
            "        if agent_instance is None:",
            f"            agent_instance = {call_symbol}(**parameters)",
            "            _AGENT_RUNTIME_CACHE[cache_key] = agent_instance",
            "        prompt_text = prompt if isinstance(prompt, str) else ''",
            "        return _atlas_execute_agent(agent_instance, prompt_text, metadata)",
            "    _atlas_require_validation()",
            f"    instance = {call_symbol}(**parameters)",
            "    return StepwiseAgentAdapter(instance)",
        ]
    )
    body_lines = [
        f"def {AGENT_FUNCTION_NAME}(**kwargs):",
        '    """Atlas-generated agent factory."""',
        *lines,
    ]
    factory_body = preamble + "\n".join(body_lines)
    internal_notes = [
        f"Generated agent factory wrapping {candidate.dotted_path()}.",
        "Bridges runtime calls to the repository agent while keeping discovery stepwise-safe.",
    ]
    return FactorySnippet(
        function_name=AGENT_FUNCTION_NAME,
        imports=imports,
        helpers=[AGENT_RUNTIME_HELPER],
        factory_body=factory_body,
        internal_notes=internal_notes,
        prerequisites=[],
        preflight=[],
        auto_skip=False,
    )


def build_function_agent_factory_snippet(candidate: Candidate) -> FactorySnippet:
    """Build factory snippet for a factory function/attribute agent.

    Handles both module-level callables and attributes. Supports dual-mode
    execution (discovery vs runtime) with agent caching.

    Args:
        candidate: Discovered agent factory candidate

    Returns:
        FactorySnippet with AGENT_RUNTIME_HELPER included
    """
    imports = ["import json", "import os", "from atlas.sdk.wrappers import StepwiseAgentAdapter"]
    if candidate.factory_kind == "attribute":
        imports.append("import importlib")
        body_lines = [
            f"def {AGENT_FUNCTION_NAME}(**kwargs):",
            '    """Atlas-generated agent factory wrapping repository attribute."""',
            "    has_prompt = 'prompt' in kwargs",
            "    has_metadata = 'metadata' in kwargs",
            "    prompt = kwargs.pop('prompt', None)",
            "    metadata = kwargs.pop('metadata', None)",
            "    runtime_invocation = has_prompt or has_metadata",
            "    parameters = dict(kwargs)",
            f"    module = importlib.import_module('{candidate.module}')",
            f"    target = getattr(module, '{candidate.qualname}')",
            "    if runtime_invocation:",
            f"        cache_key = ('agent', '{candidate.module}', '{candidate.qualname}')",
            "        agent_instance = _AGENT_RUNTIME_CACHE.get(cache_key)",
            "        if agent_instance is None:",
            "            if callable(target):",
            "                agent_instance = target(**parameters)",
            "            else:",
            "                if parameters:",
            "                    raise TypeError('Agent factory does not accept keyword arguments.')",
            "                agent_instance = target",
            "            _AGENT_RUNTIME_CACHE[cache_key] = agent_instance",
            "        prompt_text = prompt if isinstance(prompt, str) else ''",
            "        return _atlas_execute_agent(agent_instance, prompt_text, metadata)",
            "    _atlas_require_validation()",
            "    if callable(target):",
            "        instance = target(**parameters)",
            "    else:",
            "        if parameters:",
            "            raise TypeError('Agent factory does not accept keyword arguments.')",
            "        instance = target",
            "    return StepwiseAgentAdapter(instance)",
        ]
    else:
        target_symbol = candidate.qualname
        call_symbol = target_symbol
        if target_symbol == AGENT_FUNCTION_NAME:
            call_symbol = f"_repo_{AGENT_FUNCTION_NAME}"
            import_binding = f"from {candidate.module} import {target_symbol} as {call_symbol}"
        else:
            import_binding = f"from {candidate.module} import {target_symbol}"
        imports.append(import_binding)
        body_lines = [
            f"def {AGENT_FUNCTION_NAME}(**kwargs):",
            '    """Atlas-generated agent factory wrapping repository callable."""',
            "    has_prompt = 'prompt' in kwargs",
            "    has_metadata = 'metadata' in kwargs",
            "    prompt = kwargs.pop('prompt', None)",
            "    metadata = kwargs.pop('metadata', None)",
            "    runtime_invocation = has_prompt or has_metadata",
            "    parameters = dict(kwargs)",
            f"    cache_key = ('agent', '{candidate.module}', '{candidate.qualname}')",
            "    if runtime_invocation:",
            "        agent_instance = _AGENT_RUNTIME_CACHE.get(cache_key)",
            "        if agent_instance is None:",
            f"            agent_instance = {call_symbol}(**parameters)",
            "            _AGENT_RUNTIME_CACHE[cache_key] = agent_instance",
            "        prompt_text = prompt if isinstance(prompt, str) else ''",
            "        return _atlas_execute_agent(agent_instance, prompt_text, metadata)",
            "    _atlas_require_validation()",
            f"    instance = {call_symbol}(**parameters)",
            "    return StepwiseAgentAdapter(instance)",
        ]
    factory_body = "\n".join(body_lines)
    internal_notes = [
        f"Delegates to {candidate.module}:{candidate.qualname}.",
        "Bridges runtime calls to the repository agent while keeping discovery stepwise-safe.",
    ]
    return FactorySnippet(
        function_name=AGENT_FUNCTION_NAME,
        imports=imports,
        helpers=[AGENT_RUNTIME_HELPER],
        factory_body=factory_body,
        internal_notes=internal_notes,
        prerequisites=[],
        preflight=[],
        auto_skip=False,
    )


def build_environment_factory_snippet(candidate: Candidate, defaults: dict[str, object]) -> FactorySnippet:
    """Build factory snippet for an environment (dispatcher function).

    Selects appropriate builder based on whether candidate is a factory function.

    Args:
        candidate: Discovered environment candidate
        defaults: Default keyword arguments (unused for factory functions)

    Returns:
        FactorySnippet from appropriate builder
    """
    if candidate.is_factory:
        return build_function_environment_factory_snippet(candidate)
    return build_basic_environment_factory_snippet(candidate, defaults)


def build_agent_factory_snippet(candidate: Candidate, defaults: dict[str, object]) -> FactorySnippet:
    """Build factory snippet for an agent (dispatcher function).

    Selects appropriate builder based on whether candidate is a factory function.

    Args:
        candidate: Discovered agent candidate
        defaults: Default keyword arguments (unused for factory functions)

    Returns:
        FactorySnippet from appropriate builder
    """
    if candidate.is_factory:
        return build_function_agent_factory_snippet(candidate)
    return build_basic_agent_factory_snippet(candidate, defaults)
