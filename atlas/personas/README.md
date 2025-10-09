# atlas.personas

The `atlas.personas` package hosts the façade classes that encapsulate Atlas' default Student and Teacher behaviours.

## Extending personas

- **Student** (`atlas.personas.student.Student`) orchestrates planning and execution via LangGraph.
  - Override or subclass to customize tool routing, retry policies, or prompt shaping.
  - Prompts are provided through `atlas.prompts.build_student_prompts`.
- **Teacher** (`atlas.personas.teacher.Teacher`) reviews plans, validates execution, and generates guidance.
  - Supply custom `TeacherConfig` prompts or subclass to integrate additional scoring logic.

### Related modules

- `atlas.runtime.agent_loop` – LangGraph execution primitives used by the Student.
- `atlas.prompts` – helpers for generating persona prompts.
- `atlas.evaluation` – reward models referenced by the Teacher during validation.

Keep persona customisations focused on behaviour and prompt logic; persist telemetry or storage changes via the runtime packages.
