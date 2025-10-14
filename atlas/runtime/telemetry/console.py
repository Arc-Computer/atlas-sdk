from __future__ import annotations

import datetime
import sys
import threading
from typing import Any, Dict, Optional, TextIO, Tuple

from atlas.runtime.models import IntermediateStep
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.types import Result


class ConsoleTelemetryStreamer:
    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout
        self._lock = threading.Lock()
        self._subscription = None
        self._execution_context: ExecutionContext | None = None
        self._task_name = ""
        self._session_started_at: datetime.datetime | None = None
        self._attempt_starts: dict[Tuple[int, int], datetime.datetime] = {}
        self._pending_guidance: dict[Tuple[int, int], str] = {}
        self._step_attempts: dict[int, int] = {}
        self._step_names: dict[int, str] = {}
        self._execution_mode: str = "stepwise"

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
        self._write(f"=== Atlas task completed in {duration} ===")
        self._render_completion_summary(result)

    def session_failed(self, error: BaseException) -> None:
        duration = self._session_duration()
        self._write(f"=== Atlas task failed after {duration}: {error} ===")

    def _session_duration(self) -> str:
        if self._session_started_at is None:
            return "0.0s"
        delta = datetime.datetime.now() - self._session_started_at
        return f"{delta.total_seconds():.1f}s"

    def _write(self, text: str) -> None:
        with self._lock:
            print(text, file=self._output, flush=True)

    def _handle_event(self, event: IntermediateStep) -> None:
        event_type = event.event_type
        if event_type == IntermediateStepType.WORKFLOW_START:
            self._handle_workflow_start(event)
        elif event_type == IntermediateStepType.TASK_START:
            self._handle_task_start(event)
        elif event_type == IntermediateStepType.TASK_END:
            self._handle_task_end(event)

    def _handle_workflow_start(self, event: IntermediateStep) -> None:
        data = event.payload.data
        payload_input = data.input if data is not None else None
        if isinstance(payload_input, dict):
            mode = payload_input.get("execution_mode")
            if isinstance(mode, str):
                self._execution_mode = mode
        self._capture_plan_metadata()

    def _handle_task_start(self, event: IntermediateStep) -> None:
        data = event.payload.data
        payload_input = data.input if data is not None else None
        if not isinstance(payload_input, dict):
            return
        self._capture_plan_metadata()
        step_id, description = self._extract_step_info(payload_input.get("step"))
        if step_id is None:
            return
        attempt = payload_input.get("attempt")
        if not isinstance(attempt, int):
            attempt = self._step_attempts.get(step_id, 0) + 1
        self._step_attempts[step_id] = attempt
        self._step_names.setdefault(step_id, description)
        self._attempt_starts[(step_id, attempt)] = datetime.datetime.now()
        guidance_notes = payload_input.get("guidance") or []
        if isinstance(guidance_notes, list) and guidance_notes:
            last_note = guidance_notes[-1]
            if isinstance(last_note, str):
                self._pending_guidance[(step_id, attempt)] = self._shorten(last_note, 120)

    def _handle_task_end(self, event: IntermediateStep) -> None:
        step_id = self._step_id_from_event(event)
        if step_id is None:
            return
        attempt = self._step_attempts.get(step_id, 1)
        description = self._step_names.get(step_id, f"step {step_id}")
        payload_output = event.payload.data.output if event.payload.data is not None else None
        if isinstance(payload_output, dict) and "error" in payload_output:
            reason = self._shorten(str(payload_output.get("error", "")), 80)
            self._write(
                f"STEP {step_id}: {description} | actor=student | attempt={attempt} | validation=ERROR ({reason})"
            )
            return
        evaluation = self._coerce_dict(payload_output.get("evaluation") if isinstance(payload_output, dict) else {})
        validation = self._coerce_dict(evaluation.get("validation"))
        status = payload_output.get("status") if isinstance(payload_output, dict) else None
        if status is None:
            status = validation.get("status")
        valid = bool(validation.get("valid"))
        guidance_text = validation.get("guidance")
        if not isinstance(guidance_text, str):
            guidance_text = ""
        display_reason = status or ""
        if not valid:
            display_reason = guidance_text or display_reason
        reason = self._shorten(str(display_reason), 80)
        runtime_payload = self._coerce_dict(payload_output.get("runtime") if isinstance(payload_output, dict) else {})
        timings = self._coerce_dict(runtime_payload.get("timings_ms"))
        duration_ms = timings.get("total_ms")
        if duration_ms is None:
            start_time = self._attempt_starts.get((step_id, attempt))
            if start_time is not None:
                duration_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        duration_text = f"{duration_ms:.1f}ms" if isinstance(duration_ms, (int, float)) else "n/a"
        validation_label = "PASS" if valid else "FAIL"
        line = (
            f"STEP {step_id}: {description} | actor=student | attempt={attempt} | "
            f"validation={validation_label} ({reason}) | duration={duration_text}"
        )
        self._write(line)
        guidance_summary = self._pending_guidance.pop((step_id, attempt), None)
        if guidance_summary:
            self._write(
                f"STEP {step_id}: {description} | actor=teacher | attempt={attempt} | guidance={guidance_summary}"
            )
        self._attempt_starts.pop((step_id, attempt), None)

        reward_payload = self._coerce_dict(evaluation.get("reward"))
        score_value = reward_payload.get("score")
        score_text = f"{float(score_value):.2f}" if isinstance(score_value, (int, float)) else "n/a"
        judges = reward_payload.get("judges")
        rim_scores: list[str] = []
        if isinstance(judges, list):
            for index, judge in enumerate(judges, start=1):
                judge_payload = self._coerce_dict(judge)
                identifier = judge_payload.get("identifier") or f"judge{index}"
                judge_score = judge_payload.get("score")
                if isinstance(judge_score, (int, float)):
                    rim_scores.append(f"{identifier}:{float(judge_score):.2f}")
        rim_display = ", ".join(rim_scores) if rim_scores else "none"
        self._write(
            f"STEP {step_id}: retry {attempt} | Reward score={score_text} | RIM scores: {rim_display}"
        )

    def _capture_plan_metadata(self) -> None:
        if self._execution_context is None:
            return
        metadata = self._execution_context.metadata
        plan = metadata.get("plan")
        if not isinstance(plan, dict):
            return
        for entry in plan.get("steps", []):
            step_id = entry.get("id")
            description = entry.get("description") or ""
            if isinstance(step_id, int):
                self._step_names.setdefault(step_id, description)

    def _extract_step_info(self, step_payload: Any) -> Tuple[Optional[int], str]:
        if not isinstance(step_payload, dict):
            return None, ""
        step_id = step_payload.get("id")
        description = step_payload.get("description") or ""
        if isinstance(step_id, int):
            return step_id, description
        return None, description

    def _step_id_from_event(self, event: IntermediateStep) -> Optional[int]:
        name = event.payload.name or ""
        if name.startswith("step_"):
            try:
                return int(name.split("_", maxsplit=1)[1])
            except ValueError:
                return None
        return None

    def _render_completion_summary(self, result: Result) -> None:
        execution_mode = self._execution_mode
        if self._execution_context is not None:
            metadata = self._execution_context.metadata
            execution_mode = metadata.get("execution_mode", execution_mode)
            plan_payload = metadata.get("plan")
        else:
            metadata = {}
            plan_payload = None
        if execution_mode == "single_shot":
            self._write("Single-shot mode executed; step loop skipped.")
            self._render_adaptive_summary()
        else:
            self._render_plan(plan_payload)
        self._write("Final Answer:")
        final_lines = result.final_answer.splitlines() or [""]
        for line in final_lines:
            self._write(f"  {line}")
        attempt_summary, judge_calls = self._compute_metrics(result)
        runtime = self._session_duration()
        adaptive_summary = metadata.get("adaptive_summary") if isinstance(metadata, dict) else {}
        summary_line = f"Summary | execution_mode={execution_mode} | total_runtime={runtime} | judge_calls={judge_calls}"
        mode_display = adaptive_summary.get("adaptive_mode") if isinstance(adaptive_summary, dict) else None
        if isinstance(mode_display, str) and mode_display and mode_display != execution_mode:
            summary_line += f" | adaptive_mode={mode_display}"
        confidence = adaptive_summary.get("confidence") if isinstance(adaptive_summary, dict) else None
        if isinstance(confidence, (int, float)):
            summary_line += f" | adaptive_confidence={confidence:.2f}"
        certification_flag = adaptive_summary.get("certification_run") if isinstance(adaptive_summary, dict) else None
        if certification_flag:
            summary_line += " | certification=True"
        self._write(summary_line)
        if attempt_summary:
            self._write(f"  attempts: {attempt_summary}")
        reward_line = self._format_reward_highlight(metadata)
        if reward_line:
            self._write(f"  {reward_line}")
        learning_lines = self._collect_learning_highlights(metadata)
        for label, value in learning_lines:
            self._write(f"  {label}: {self._shorten(value, 160)}")

    def _render_plan(self, plan_payload: Any) -> None:
        if not isinstance(plan_payload, dict):
            return
        steps = plan_payload.get("steps") or []
        if not steps:
            return
        self._write(f"Plan ready ({len(steps)} steps):")
        for entry in steps:
            if not isinstance(entry, dict):
                continue
            step_id = entry.get("id")
            description = entry.get("description") or ""
            if isinstance(step_id, int):
                self._write(f"  {step_id}. {description}")
        self._render_adaptive_summary()

    def _render_adaptive_summary(self) -> None:
        if self._execution_context is None:
            return
        adaptive = self._execution_context.metadata.get("adaptive_summary") or {}
        if not isinstance(adaptive, dict):
            return
        mode = adaptive.get("adaptive_mode")
        confidence = adaptive.get("confidence")
        certification = adaptive.get("certification_run")
        probe = adaptive.get("probe") if isinstance(adaptive.get("probe"), dict) else None
        line_parts = ["Adaptive:"]
        if mode:
            line_parts.append(f"mode={mode}")
        if confidence is not None:
            line_parts.append(f"confidence={confidence}")
        if certification:
            line_parts.append("certification=True")
        self._write(" ".join(line_parts))
        if probe:
            evidence = probe.get("evidence") if isinstance(probe.get("evidence"), list) else []
            probe_line = (
                f"  probe -> mode={probe.get('mode')} confidence={probe.get('confidence')}"
            )
            self._write(probe_line)
            if evidence:
                self._write(f"  probe evidence: {', '.join(evidence)}")
        notes = adaptive.get("mode_history")
        if isinstance(notes, list) and notes:
            summary = "; ".join(
                f"{entry.get('mode')}({entry.get('reason', 'n/a')})" if isinstance(entry, dict) else str(entry)
                for entry in notes[-3:]
            )
            self._write(f"  recent decisions: {summary}")

    def _format_reward_highlight(self, metadata: Dict[str, Any]) -> str | None:
        reward_payload = metadata.get("session_reward")
        if isinstance(reward_payload, dict):
            score = reward_payload.get("score")
            if isinstance(score, (int, float)):
                rationale = reward_payload.get("rationale")
                message = f"Reward score={score:.2f}"
                if isinstance(rationale, str) and rationale.strip():
                    message += f" ({self._shorten(rationale.strip(), 80)})"
                return message
        summary = metadata.get("reward_summary")
        if isinstance(summary, dict):
            score = summary.get("score")
            if isinstance(score, (int, float)):
                return f"Reward score={score:.2f}"
            average = summary.get("average")
            if isinstance(average, (int, float)):
                count = summary.get("count")
                if isinstance(count, int):
                    return f"Reward average={average:.2f} across {count} steps"
                return f"Reward average={average:.2f}"
        return None

    def _collect_learning_highlights(self, metadata: Dict[str, Any]) -> list[tuple[str, str]]:
        highlights: list[tuple[str, str]] = []
        student_learning = metadata.get("student_learning")
        if isinstance(student_learning, str) and student_learning.strip():
            highlights.append(("student_learning", student_learning.strip()))
        teacher_learning = metadata.get("teacher_learning")
        if isinstance(teacher_learning, str) and teacher_learning.strip():
            highlights.append(("teacher_learning", teacher_learning.strip()))
        return highlights

    def _compute_metrics(self, result: Result) -> Tuple[str, int]:
        attempt_counts: dict[int, int] = {}
        for step_result in result.step_results:
            attempt_counts[step_result.step_id] = step_result.attempts
        steps_meta = {}
        if self._execution_context is not None:
            steps_meta = self._execution_context.metadata.get("steps", {}) or {}
        if not attempt_counts and steps_meta:
            for step_id, meta in steps_meta.items():
                attempts = len(meta.get("attempts", []))
                if attempts:
                    attempt_counts[step_id] = attempts
        judge_calls = 0
        for meta in steps_meta.values():
            for attempt in meta.get("attempts", []):
                if not attempt.get("reward_skipped", False):
                    judge_calls += 1
        attempt_summary = ", ".join(
            f"{step_id}={count}" for step_id, count in sorted(attempt_counts.items())
        )
        return attempt_summary, judge_calls

    def _coerce_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    def _shorten(self, text: str, limit: int = 80) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"
