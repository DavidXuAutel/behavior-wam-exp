"""R1Pro action-space adapter stub (Task 3/4).

Fill action_dim and conventions after inspecting OmniGibson r1pro.yaml.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class R1ProActionAdapter:
    def __init__(self, action_dim: int | None = None) -> None:
        self.action_dim = action_dim

    def to_env(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_dim is not None and arr.size != self.action_dim:
            raise ValueError(
                f"action dim {arr.size} != expected {self.action_dim}"
            )
        return arr
