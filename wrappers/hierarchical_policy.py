"""Hierarchical policy stub (Task 4).

Planner proposes next skill (eval-legal completion only).
MoT-WAM executor returns an action chunk; router splits nav vs manip.
"""

from __future__ import annotations

from typing import Any, Protocol


NAV_SKILL_PREFIXES = ("move to", "navigate", "push to")


class SkillPlanner(Protocol):
    def next_skill(self, obs: dict[str, Any], task: dict[str, Any]) -> str: ...


class SkillExecutor(Protocol):
    def act(self, obs: dict[str, Any], skill: str) -> Any: ...


def skill_class(skill: str) -> str:
    s = skill.lower().strip()
    if any(s.startswith(p) or p in s for p in NAV_SKILL_PREFIXES):
        return "navigation"
    return "manipulation"


class HierarchicalPolicy:
    def __init__(self, planner: SkillPlanner, executor: SkillExecutor) -> None:
        self.planner = planner
        self.executor = executor
        self._current_skill: str | None = None
        self._skill_steps = 0

    def reset(self, task_info: dict[str, Any]) -> None:
        self._current_skill = None
        self._skill_steps = 0
        self._task = task_info

    def act(self, obs: dict[str, Any]) -> Any:
        if "depth" not in obs:
            raise ValueError("depth is required in observation pack")
        task = getattr(self, "_task", {})
        skill = self.planner.next_skill(obs, task)
        self._current_skill = skill
        self._skill_steps += 1
        action = self.executor.act(obs, skill)
        # Router hint for downstream adapters / filters
        if isinstance(action, dict):
            action = {**action, "skill_class": skill_class(skill)}
        return action
