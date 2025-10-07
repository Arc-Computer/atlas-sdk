from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union


class Step(BaseModel):
    id: int
    description: str
    tool: Optional[str] = None
    tool_params: Optional[dict] = None
    depends_on: List[Union[int, str]] = Field(default_factory=list)


class Plan(BaseModel):
    steps: List[Step]


class StepResult(BaseModel):
    step_id: int
    trace: str
    output: str
    evaluation: dict
    attempts: int = 1
    metadata: dict = Field(default_factory=dict)


class Result(BaseModel):
    final_answer: str
    plan: Plan
    step_results: List[StepResult]
