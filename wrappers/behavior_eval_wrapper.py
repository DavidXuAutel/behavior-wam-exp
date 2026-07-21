"""OmniGibson evaluation wrapper stub (Task 2).

Implement the official challenge eval interface here.
Policy code lives entirely in this repo (independent WAM).
"""

from __future__ import annotations

from typing import Any


class BehaviorEvalWrapper:
    """Thin adapter between hierarchical policy and OmniGibson evaluator."""

    def __init__(self, policy: Any, robot_config_path: str | None = None) -> None:
        self.policy = policy
        self.robot_config_path = robot_config_path

    def reset(self, task_info: dict[str, Any]) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset(task_info)

    def act(self, obs: dict[str, Any]) -> Any:
        """obs must only contain challenge-legal modalities."""
        return self.policy.act(obs)
