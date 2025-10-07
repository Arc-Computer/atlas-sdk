from __future__ import annotations

import datetime
import json
import sys
import threading
from typing import Any
from typing import Iterable
from typing import TextIO

from atlas.data_models.intermediate_step import IntermediateStep
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.orchestration.execution_context import ExecutionContext
from atlas.types import Result


class ConsoleTelemetryStreamer:
    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout
        self._lock = threading.Lock()
        self._subscription = None
        self._execution_context: ExecutionContext | None = None
        self._task_name = ""
        self._session_started_at: datetime.datetime | None = None
        self._step_guidance: dict[int, list[str]] = {}
        self._step_attempts: dict[int, int] = {}
        self._step_names: dict[int, str] = {}
        self._plan_rendered = False

    def attach(self, execution_context: ExecutionContext) -> None:
        self.detach()
        self._execution_context = execution_context
        manager = execution_context.intermediate_step_manager
        self._subscription = manager.subscribe(self._handle_event)

    def detach(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    def session_started(self, task_name: str) -> None:
        self._task_name = task_name
        self._session_started_at = datetime.datetime.now()
        timestamp = self._session_started_at.strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"=== Atlas task started: {task_name} ({timestamp}) ===")

    def session_completed(self, result: Result) -> None:
        duration = self._session_duration()
        status_line = f"=== Atlas task completed in {duration} ==="
        self._write(status_line)
        self._render_summary(result)

    def session_failed(self, error: BaseException) -> None:
        duration = self._session_duration()
        status_line = f"=== Atlas task failed after {duration}: {error} ==="
        self._write(status_line)

    def _session_duration(self) -> str:
        if self._session_started_at is None:
            return "0s"
        delta = datetime.datetime.now() - self._session_started_at
        seconds = delta.total_seconds()
        return f"{seconds:.1f}s"

    def _write(self, text: str) -> None:
        with self._lock:
            print(text, file=self._output, flush=True)

    def _coerce_evaluation(self, evaluation: Any) -> dict[str, Any]:
        if hasattr(evaluation, "to_dict"):
            return evaluation.to_dict()
        if isinstance(evaluation, dict):
            return evaluation
        return {}

    def _render_summary(self, result: Result) -> None:
        self._write("Final answer:")
        for line in result.final_answer.splitlines() or [""]:
            self._write(f"  {line}")
        rim_scores = []
        for step in result.step_results:
            evaluation = self._coerce_evaluation(step.evaluation)
            reward = evaluation.get("reward")
            score = reward.get("score") if isinstance(reward, dict) else None
            if isinstance(score, (int, float)):
                rim_scores.append(score)
            attempts = step.attempts
            label = self._step_names.get(step.step_id) or f"step {step.step_id}"
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
            self._write(f"- {label} | attempts: {attempts} | score: {score_text}")
        if rim_scores:
            best = max(rim_scores)
            avg = sum(rim_scores) / len(rim_scores)
            self._write(f"RIM scores | max: {best:.2f} | avg: {avg:.2f}")

    def _handle_event(self, event: IntermediateStep) -> None:
        lines = self._render_event(event)
        for line in lines:
            self._write(line)

    def _maybe_render_plan(self) -> list[str]:
        if self._plan_rendered and self._execution_context is not None:
            return []
        if self._execution_context is None:
            return []
        metadata = self._execution_context.metadata
        plan = metadata.get("plan")
        if not plan:
            return []
        steps = plan.get("steps") or []
        if not steps:
            return []
        self._plan_rendered = True
        lines = ["Plan ready with steps:"]
        for entry in steps:
            step_id = entry.get("id")
            description = entry.get("description") or ""
            if isinstance(step_id, int):
                self._step_names.setdefault(step_id, description)
                lines.append(f"  {step_id}. {description}")
        return lines

    def _extract_step_info(self, payload: dict[str, Any]) -> tuple[int | None, str]:
        step = payload.get("step")
        if isinstance(step, dict):
            step_id = step.get("id")
            if isinstance(step_id, int):
                description = step.get("description") or ""
                if description:
                    self._step_names.setdefault(step_id, description)
                return step_id, description
        return None, ""

    def _render_guidance(self, step_id: int | None, guidance: Iterable[str]) -> list[str]:
        if step_id is None:
            return []
        notes = list(guidance)
        previous = self._step_guidance.get(step_id, [])
        new_notes = [note for note in notes if note not in previous]
        self._step_guidance[step_id] = notes
        return [f"  guidance: {note}" for note in new_notes]

    def _format_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _shorten(self, text: str, limit: int = 120) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _render_event(self, event: IntermediateStep) -> list[str]:
        data = event.payload.data
        payload_input = data.input if data is not None else None
        payload_output = data.output if data is not None else None
        event_type = event.event_type
        lines: list[str] = []
        if event_type == IntermediateStepType.WORKFLOW_START:
            lines.append("[workflow] orchestration started")
            lines.extend(self._maybe_render_plan())
        elif event_type == IntermediateStepType.WORKFLOW_END:
            if payload_output:
                text = self._format_json(payload_output)
                lines.append(f"[workflow] completed: {self._shorten(text)}")
        elif event_type == IntermediateStepType.TASK_START:
            input_payload = payload_input if isinstance(payload_input, dict) else {}
            step_id, description = self._extract_step_info(input_payload)
            attempt = input_payload.get("attempt")
            if step_id is not None and isinstance(attempt, int):
                self._step_attempts[step_id] = attempt
            plan_lines = self._maybe_render_plan()
            lines.extend(plan_lines)
            label = description or (f"step_{step_id}" if step_id is not None else "step")
            stored_label = self._step_names.get(step_id, label) if step_id is not None else label
            attempt_text = f"attempt {attempt}" if isinstance(attempt, int) else "attempt"
            if isinstance(attempt, int) and attempt > 1:
                attempt_text = f"retry {attempt}"
            prefix = f"[step {step_id}]" if step_id is not None else "[step]"
            lines.append(f"{prefix} {attempt_text} started: {stored_label}")
            guidance_lines = self._render_guidance(step_id, input_payload.get("guidance") or [])
            lines.extend(guidance_lines)
        elif event_type == IntermediateStepType.TASK_END:
            step_id = None
            if event.payload.name and event.payload.name.startswith("step_"):
                try:
                    step_id = int(event.payload.name.split("_", maxsplit=1)[1])
                except ValueError:
                    step_id = None
            if step_id is not None:
                label = self._step_names.get(step_id)
            else:
                label = None
            if not label:
                label = event.payload.name or "step"
            summary = self._summarise_step_result(payload_output)
            lines.append(f"[step {step_id}] completed: {label}".strip())
            lines.extend(summary)
        elif event_type == IntermediateStepType.TOOL_START:
            name = event.payload.name or "tool"
            arguments = self._format_json(payload_input) if payload_input is not None else ""
            lines.append(f"[tool] {name} call -> {self._shorten(arguments)}")
        elif event_type == IntermediateStepType.TOOL_END:
            name = event.payload.name or "tool"
            result_text = self._format_json(payload_output) if payload_output is not None else ""
            lines.append(f"[tool] {name} result <- {self._shorten(result_text)}")
        else:
            generic = event.payload.name or event_type.value
            lines.append(f"[{event_type.value.lower()}] {generic}")
        return [line for line in lines if line]

    def _summarise_step_result(self, payload_output: Any) -> list[str]:
        if not isinstance(payload_output, dict):
            return []
        if "error" in payload_output:
            return [f"  error: {payload_output['error']}"]
        lines: list[str] = []
        evaluation = payload_output.get("evaluation")
        evaluation_dict = self._coerce_evaluation(evaluation)
        if evaluation_dict:
            validation = evaluation_dict.get("validation")
            if isinstance(validation, dict) and "valid" in validation:
                status = "approved" if validation.get("valid") else "rejected"
                lines.append(f"  validation: {status}")
                rationale = validation.get("rationale")
                if rationale:
                    lines.append(f"  rationale: {self._shorten(str(rationale))}")
            reward = evaluation_dict.get("reward")
            if isinstance(reward, dict):
                score = reward.get("score")
                if isinstance(score, (int, float)):
                    lines.append(f"  reward score: {score:.2f}")
                uncertainty = reward.get("uncertainty")
                if isinstance(uncertainty, (int, float)):
                    lines.append(f"  reward uncertainty: {uncertainty:.2f}")
                judges = reward.get("judges")
                if isinstance(judges, list) and judges:
                    lines.append(f"  judges: {len(judges)}")
        output_text = payload_output.get("output")
        if output_text:
            formatted = self._format_json(output_text)
            lines.append(f"  output: {self._shorten(formatted)}")
        return lines
