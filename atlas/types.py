from typing import Any, List, Optional, Union
from typing import Literal

from pydantic import BaseModel, Field
from pydantic import ConfigDict
from pydantic import model_serializer
from pydantic import model_validator

from atlas.runtime.schema import AtlasRewardBreakdown


class Step(BaseModel):
    id: int
    description: str
    tool: Optional[str] = None
    tool_params: Optional[dict] = None
    depends_on: List[Union[int, str]] = Field(default_factory=list)


class Plan(BaseModel):
    steps: List[Step]
    execution_mode: Literal["stepwise", "single_shot"] = "stepwise"


class StepEvaluation(BaseModel):
    """Structured evaluation output for a plan step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    validation: dict[str, Any] = Field(default_factory=dict)
    reward: AtlasRewardBreakdown

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            validation = value.get("validation") or {}
            reward_payload = value.get("reward") or {}
            if isinstance(reward_payload, AtlasRewardBreakdown):
                reward = reward_payload
            else:
                reward = AtlasRewardBreakdown.from_dict(reward_payload if isinstance(reward_payload, dict) else {})
            return {"validation": validation, "reward": reward}
        raise TypeError("Unsupported evaluation payload")

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data["reward"] = self.reward.to_dict()
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation": self.validation,
            "reward": self.reward.to_dict(),
        }


class StepResult(BaseModel):
    step_id: int
    trace: str
    output: str
    evaluation: StepEvaluation
    attempts: int = 1
    metadata: dict = Field(default_factory=dict)


class Result(BaseModel):
    final_answer: str
    plan: Plan
    step_results: List[StepResult]
