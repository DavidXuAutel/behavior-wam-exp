"""Hierarchical policy stub (Task 4).

Planner proposes next skill; in-repo WAM executor returns an action chunk.
"""

from __future__ import annotations

from typing import Any, Protocol


class SkillPlanner(Protocol):
    def next_skill(self, obs: dict[str, Any], task: dict[str, Any]) -> str: ...


class SkillExecutor(Protocol):
    def act(self, obs: dict[str, Any], skill: str) -> Any: ...


class HierarchicalPolicy:
    def __init__(self, planner: SkillPlanner, executor: SkillExecutor) -> None:
        self.planner = planner
        self.executor = executor
        self._current_skill: str | None = None

    def reset(self, task_info: dict[str, Any]) -> None:
        self._current_skill = None
        self._task = task_info

    def act(self, obs: dict[str, Any]) -> Any:
        task = getattr(self, "_task", {})
        skill = self.planner.next_skill(obs, task)
        self._current_skill = skill
        return self.executor.act(obs, skill)
