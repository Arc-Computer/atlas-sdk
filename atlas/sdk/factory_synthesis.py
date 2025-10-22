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

RoleLiteral = str  # alias for readability

ENV_FUNCTION_NAME = "create_environment"
AGENT_FUNCTION_NAME = "create_agent"
GENERATED_MODULE = ".atlas.generated_factories"
ENV_VALIDATE_FLAG = "ATLAS_DISCOVERY_VALIDATE"


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
    """Structured response from the LLM for a single factory."""

    function_name: str
    imports: list[str] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)
    factory_body: str = ""
    notes: list[str] = field(default_factory=list)
    preflight: list[str] = field(default_factory=list)
    auto_skip: bool = False


@dataclass(slots=True)
class SynthesisOutcome:
    """Aggregated synthesis result for environment/agent."""

    environment_factory: tuple[str, str] | None = None
    agent_factory: tuple[str, str] | None = None
    preflight_notes: list[str] = field(default_factory=list)
    auxiliary_notes: list[str] = field(default_factory=list)
    auto_skip: bool = False


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
        self._context_cache: dict[RoleLiteral, ClassContext] = {}
        self._snippet_cache: dict[RoleLiteral, FactorySnippet] = {}
        self._error_history: dict[RoleLiteral, list[str]] = {}

    def synthesise(
        self,
        *,
        environment: Candidate | None,
        agent: Candidate | None,
        environment_kwargs: dict[str, Any],
        agent_kwargs: dict[str, Any],
    ) -> SynthesisOutcome:
        outcome = SynthesisOutcome()
        snippets: dict[str, FactorySnippet] = {}

        if environment is not None:
            env_needed, env_context = self._analyse_candidate(environment, environment_kwargs)
            if env_needed:
                snippet = self._generate_snippet("environment", env_context, previous_error=None)
                snippets["environment"] = snippet
                self._snippet_cache["environment"] = snippet
                outcome.environment_factory = (GENERATED_MODULE, snippet.function_name)
                outcome.preflight_notes.extend(snippet.preflight)
                outcome.auxiliary_notes.extend(snippet.notes)
                outcome.auto_skip = outcome.auto_skip or snippet.auto_skip

        if agent is not None:
            agent_needed, agent_context = self._analyse_candidate(agent, agent_kwargs)
            if agent_needed:
                snippet = self._generate_snippet("agent", agent_context, previous_error=None)
                snippets["agent"] = snippet
                self._snippet_cache["agent"] = snippet
                outcome.agent_factory = (GENERATED_MODULE, snippet.function_name)
                outcome.preflight_notes.extend(snippet.preflight)
                outcome.auxiliary_notes.extend(snippet.notes)
                outcome.auto_skip = outcome.auto_skip or snippet.auto_skip

        if snippets:
            self._emit_module(snippets)

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
            self._emit_module(snippets)

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

    def _analyse_candidate(
        self,
        candidate: Candidate,
        provided_kwargs: dict[str, Any],
    ) -> tuple[bool, ClassContext]:
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
        self._context_cache[candidate.role] = context
        missing = [arg for arg in required_args if arg not in provided_kwargs]
        return (len(missing) > 0), context

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

    def _generate_snippet(
        self,
        role: RoleLiteral,
        context: ClassContext,
        *,
        previous_error: str | None,
    ) -> FactorySnippet:
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
                    overrides={"temperature": 0.2, "max_tokens": 2048},
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
    def _default_models() -> list[LLMParameters]:
        return [
            LLMParameters(
                provider=LLMProvider.ANTHROPIC,
                model="claude-haiku-4-5",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.1,
                max_output_tokens=2048,
                timeout_seconds=60.0,
            ),
            LLMParameters(
                provider=LLMProvider.GEMINI,
                model="gemini/gemini-2.5-flash",
                api_key_env="GEMINI_API_KEY",
                temperature=0.1,
                max_output_tokens=1536,
                timeout_seconds=60.0,
            ),
        ]


_SYNTHESIS_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are assisting Atlas CLI in generating Python factory helpers for autodiscovery.
    Each factory MUST:
    - Import the target class directly from its module path.
    - Provide deterministic defaults for required constructor arguments using the context provided.
    - Accept arbitrary keyword overrides and merge them with the defaults.
    - Delay side effects (network, Docker, database) until the environment variable {ENV_VALIDATE_FLAG} is "1".
      When validation is not enabled, raise RuntimeError with a useful message and include any preflight guidance
      in the generated metadata.
    - Avoid importing heavyweight modules unless validate is true (wrap them inside the validate branch).
    - Return a fully constructed instance when validation is enabled.

    Respond with strict JSON (no markdown fences) containing:
      - "function_name": name of the factory function to emit (snake_case).
      - "imports": array of import statements required at module scope (strings).
      - "helpers": array of helper function/constant definitions (each as a single string) or an empty array.
      - "factory": the full function definition implementing the factory.
      - "notes": array of short textual notes about assumptions or TODOs (may be empty).
      - "preflight": array of warnings/preflight steps to surface to the user (may be empty).
      - "auto_skip": boolean indicating if discovery should auto-skip running the loop until validation.

    Ensure the resulting code is valid Python 3.12 and uses typing hints where reasonable.
    """
).strip()
