"""Dependency graph utilities for validating and ordering plan steps."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set

from atlas.types import Plan
from atlas.types import Step


class DependencyGraph:
    def __init__(self, plan: Plan) -> None:
        self._steps: Dict[int, Step] = {step.id: step for step in plan.steps}
        self._edges: Dict[int, Set[int]] = defaultdict(set)
        self._reverse_edges: Dict[int, Set[int]] = defaultdict(set)
        self._populate_edges(plan.steps)

    def _populate_edges(self, steps: Iterable[Step]) -> None:
        for step in steps:
            for dependency in self._normalise_dependency_list(step.depends_on):
                if dependency not in self._steps:
                    raise ValueError(f"Step {step.id} depends on unknown step {dependency}")
                self._edges[dependency].add(step.id)
                self._reverse_edges[step.id].add(dependency)
            self._edges.setdefault(step.id, set())
            self._reverse_edges.setdefault(step.id, set())

    def _normalise_dependency_list(self, dependencies: Iterable[int | str]) -> List[int]:
        result: List[int] = []
        for item in dependencies:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str) and item.startswith("#"):
                try:
                    result.append(int(item[1:]))
                except ValueError as exc:
                    raise ValueError(f"Invalid dependency placeholder '{item}'") from exc
            else:
                raise ValueError(f"Unsupported dependency value '{item}'")
        return result

    def has_cycles(self) -> bool:
        try:
            self.topological_levels()
            return False
        except ValueError:
            return True

    def topological_levels(self) -> List[List[int]]:
        in_degree = {node: len(deps) for node, deps in self._reverse_edges.items()}
        ready = [node for node, degree in in_degree.items() if degree == 0]
        levels: List[List[int]] = []
        visited = 0
        while ready:
            current_level = sorted(ready)
            levels.append(current_level)
            new_ready: List[int] = []
            for node in current_level:
                visited += 1
                for child in self._edges[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        new_ready.append(child)
            ready = new_ready
        if visited != len(self._steps):
            raise ValueError("Plan contains circular dependencies")
        return levels
