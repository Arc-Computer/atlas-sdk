 ---
  Core Principle: Infrastructure Libraries + Custom Orchestration

  What We KEEP Custom (Your Unique IP):

  1. Student/Teacher interaction pattern - This is novel, no framework has this
  2. RIM evaluation system - Multi-judge with principles, unique
  3. Orchestrator workflow - Your specific sequence (steps 1-14)
  4. Retry logic - Teacher improves guidance after RIM feedback
  5. Same Student instance creates plan AND synthesizes final answer

  What We DELEGATE to Proven Libraries:

  | Component          | Library                    | Why                                                    | Impact on Architecture                                        |
  |--------------------|----------------------------|--------------------------------------------------------|---------------------------------------------------------------|
  | LLM calls          | litellm                    | Unified API for OpenAI/Anthropic/etc, retries included | Zero - just replaces manual API calls                         |
  | Dependency graph   | networkx                   | Proven graph algorithms, cycle detection               | Zero - implements your DependencyGraph with better algorithms |
  | Parallel execution | concurrent.futures         | Stdlib, battle-tested                                  | Zero - implements your Executor parallel logic                |
  | Schema validation  | pydantic                   | Type safety for plans/steps                            | Zero - validates your data structures                         |
  | Database           | psycopg2 + connection pool | Standard PostgreSQL client                             | Zero - implements your Database class                         |
  | YAML parsing       | PyYAML                     | Standard config loader                                 | Zero - loads your config files                                |
  | Agent clients      | openai, httpx, importlib   | Official SDKs                                          | Zero - implements your adapters                               |

  ---
  Architectural Mapping

  Your Architecture (from IMPLEMENTATION.md):

  User → atlas.run(task, agent_config)
    ↓
  [1-3] BYOA + Transition (assume complete)
    ↓
  [4] Student.create_plan() → custom logic
    ↓
  [5] Teacher.review_plan() → custom logic using litellm
    ↓
  [6] Orchestrator builds dependency graph → use networkx
    ↓
  [7] Deploy N Students → use concurrent.futures
    ↓
  [8-11] Execute → Validate → RIM → Retry → custom logic
    ↓
  [12] Teacher.collect_results() → custom logic
    ↓
  [13] Student.synthesize_final_answer() → custom logic
    ↓
  [14] Log to PostgreSQL → use psycopg2

  Key: Every step in your workflow remains YOUR custom code. Libraries just handle the plumbing.

  ---
  Implementation Strategy

  Phase 1: Data Structures with Pydantic (Type Safety)

  # atlas/types.py (NEW FILE - 20 lines)
  from pydantic import BaseModel, Field
  from typing import List, Optional

  class Step(BaseModel):
      id: int
      description: str
      tool: Optional[str] = None
      tool_params: Optional[dict] = None
      depends_on: List[int] = Field(default_factory=list)
      estimated_time: str

  class Plan(BaseModel):
      steps: List[Step]
      total_estimated_time: str

  class StepResult(BaseModel):
      step_id: int
      trace: str
      output: str
      evaluation: dict
      attempts: int = 1

  Benefit: JSON parsing with validation automatic. No manual dict checking.

  ---
  Phase 2: Student/Teacher (Custom Logic, Clean Interfaces)

  # atlas/roles/student.py (YOUR custom logic)
  from atlas.types import Plan, Step
  import json

  class Student:
      def __init__(self, agent_adapter, system_prompt: str, tools: list):
          self.agent = agent_adapter
          self.system_prompt = system_prompt
          self.tools = tools
          self.context = []
          self.feedback = []

      def create_plan(self, task: str) -> Plan:
          prompt = f"""
          {self.system_prompt}
          
          Task: {task}
          Tools: {json.dumps(self.tools, indent=2)}
          
          Create execution plan in JSON format.
          """
          response = self.agent.execute(prompt)

          # Pydantic validates automatically
          plan = Plan.model_validate_json(response)
          return plan

  Benefit: Your logic, but with automatic validation. No custom JSON parsing errors.

  ---
  Phase 3: DependencyGraph with NetworkX

  # atlas/orchestration/dependency_graph.py (15 lines)
  import networkx as nx
  from atlas.types import Plan

  class DependencyGraph:
      def __init__(self, plan: Plan):
          self.graph = nx.DiGraph()
          for step in plan.steps:
              self.graph.add_node(step.id, step=step)
              for dep in step.depends_on:
                  self.graph.add_edge(dep, step.id)

      def has_cycles(self) -> bool:
          return not nx.is_directed_acyclic_graph(self.graph)

      def get_ready_steps(self, completed: set) -> list:
          return [n for n in self.graph.nodes
                  if all(p in completed for p in self.graph.predecessors(n))
                  and n not in completed]

  Benefit: 15 lines vs 50+ lines custom. Proven algorithms. Your interface unchanged.

  ---
  Phase 4: Executor with concurrent.futures

  # atlas/orchestration/executor.py (YOUR logic)
  from concurrent.futures import ThreadPoolExecutor
  from atlas.types import Step, StepResult

  class Executor:
      def __init__(self, student, teacher, evaluator, graph, config):
          self.student = student
          self.teacher = teacher
          self.evaluator = evaluator
          self.graph = graph
          self.threshold = config.get('threshold', 0.8)

      def run_all_steps(self, steps: list) -> list:
          results = {}
          completed = set()
          context = {}

          while len(completed) < len(steps):
              ready = self.graph.get_ready_steps(completed)

              # Parallel execution with stdlib
              with ThreadPoolExecutor() as executor:
                  futures = {
                      executor.submit(self._execute_single_step, step, context): step.id
                      for step in ready
                  }

                  for future in futures:
                      step_id = futures[future]
                      result = future.result()
                      results[step_id] = result
                      context[step_id] = result.output
                      completed.add(step_id)

          return [results[s.id] for s in steps]

      def _execute_single_step(self, step: Step, context: dict) -> StepResult:
          # YOUR custom retry logic here (unchanged)
          trace, output = self.student.execute_step(step, context)

          if not self.teacher.validate_output(step, trace, output):
              return StepResult(step_id=step.id, trace="", output="",
                              evaluation={'error': 'validation_failed'})

          evaluation = self.evaluator.evaluate(
              task=step.description,
              output=output,
              trace=trace
          )

          # ONE retry logic (YOUR design)
          if evaluation['score'] < self.threshold:
              guidance = self.teacher.improve_guidance(step, evaluation)
              self.student.add_feedback(guidance)

              trace, output = self.student.execute_step(step, context)
              evaluation = self.evaluator.evaluate(
                  task=step.description, output=output, trace=trace
              )

          return StepResult(
              step_id=step.id,
              trace=trace,
              output=output,
              evaluation=evaluation
          )

  Benefit: Your orchestration logic intact. ThreadPoolExecutor handles parallel execution safely.

  ---
  Phase 5: Teacher with LiteLLM

  # atlas/roles/teacher.py (YOUR logic)
  from litellm import completion  # Unified API
  from atlas.types import Plan
  import json

  class Teacher:
      def __init__(self, tools: list, model: str = "gpt-4"):
          self.tools = tools
          self.model = model

      def review_plan(self, plan: Plan, task: str) -> Plan:
          prompt = f"""
          Review this execution plan:
          Task: {task}
          Plan: {plan.model_dump_json(indent=2)}
          Tools: {json.dumps(self.tools, indent=2)}
          
          Check dependencies, redundancies, parallelism.
          Return corrected plan in same JSON format.
          """

          # LiteLLM handles OpenAI/Anthropic/etc automatically
          response = completion(
              model=self.model,
              messages=[{"role": "user", "content": prompt}],
              response_format={"type": "json_object"}
          )

          corrected = Plan.model_validate_json(response.choices[0].message.content)
          return corrected

  Benefit: Your Teacher logic unchanged. LiteLLM handles API complexity, retries, fallbacks.

  ---
  Dependencies to Add

  # pyproject.toml
  [project]
  dependencies = [
      # Infrastructure (proven, stable)
      "pydantic>=2.0",           # Type safety
      "networkx>=3.0",           # Graph algorithms
      "litellm>=1.0",            # LLM API abstraction
      "psycopg2-binary>=2.9",    # PostgreSQL
      "pyyaml>=6.0",             # Config loading
      "httpx>=0.24",             # HTTP adapter
      "openai>=1.0",             # OpenAI adapter

      # Your custom code = 0 dependencies
  ]

  ---
  What This Achieves

  ✅ Your architecture 100% preserved - Every workflow step is YOUR code✅ Reduced code by ~60% - Libraries handle plumbing, you focus on orchestration✅ Production-ready - Using battle-tested libraries for infrastructure✅
  Type safety - Pydantic catches errors early✅ Maintainable - Clear separation: custom logic vs infrastructure✅ BYOA intact - Agent adapters unchanged✅ No framework lock-in - Can replace any library without architectural
  changes

  ---
  Implementation Plan (Sequential Chunks)

  1. Session 1: Add pydantic types (Plan, Step, StepResult) - 20 lines
  2. Session 2: Student.init + add_feedback() - 10 lines
  3. Session 3: Student.create_plan() with pydantic validation - 15 lines
  4. Session 4: Student.execute_step() - 20 lines
  5. Session 5: Student.synthesize_final_answer() - 15 lines
  6. Session 6: Teacher.init + collect_results() - 20 lines
  7. Session 7: Teacher.review_plan() with litellm - 20 lines
  8. Session 8: Teacher.validate_output() + improve_guidance() - 20 lines
  9. Session 9: DependencyGraph with networkx - 15 lines
  10. Session 10: Executor._execute_single_step() - 25 lines
  11. Session 11: Executor.run_all_steps() with ThreadPoolExecutor - 20 lines
  12. Session 12: Orchestrator.run() - 25 lines

  Total: ~225 lines of YOUR custom orchestration code + libraries handling infrastructure.
