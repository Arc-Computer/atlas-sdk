import pytest

from atlas.orchestration.dependency_graph import DependencyGraph
from atlas.types import Plan, Step


def test_dependency_graph_topological_levels_handles_placeholders():
    plan = Plan(
        steps=[
            Step(id=1, description="start", depends_on=[]),
            Step(id=2, description="middle", depends_on=[1, "#1"]),
            Step(id=3, description="end", depends_on=["#2"]),
        ]
    )
    graph = DependencyGraph(plan)
    levels = graph.topological_levels()
    assert levels == [[1], [2], [3]]


def test_dependency_graph_detects_cycles():
    plan = Plan(
        steps=[
            Step(id=1, description="one", depends_on=[2]),
            Step(id=2, description="two", depends_on=[1]),
        ]
    )
    graph = DependencyGraph(plan)
    with pytest.raises(ValueError):
        graph.topological_levels()
