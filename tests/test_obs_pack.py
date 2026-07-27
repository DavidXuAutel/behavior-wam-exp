from __future__ import annotations

import numpy as np
import pytest

from wam.obs_pack import MissingDepthError, pack_obs


def _rgb(h: int = 256, w: int = 256) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _depth(h: int = 256, w: int = 256) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32)


def test_missing_depth_raises() -> None:
    raw = {
        "rgb": _rgb(),
        "proprio": np.zeros(23, dtype=np.float32),
        "language": "pick up the cup",
    }
    with pytest.raises(MissingDepthError):
        pack_obs(raw)


def test_packed_shapes_match_lock_resolution() -> None:
    raw = {
        "rgb": _rgb(256, 256),
        "depth": _depth(256, 256),
        "proprio": np.zeros(23, dtype=np.float32),
        "language": "move to the table",
    }
    packed = pack_obs(raw, rgb_resolution=256)
    assert packed["rgb"].shape == (256, 256, 3)
    assert packed["depth"].shape == (256, 256)
    assert packed["proprio"].shape == (23,)
    assert isinstance(packed["language"], str)
    assert packed["language"]


def test_dict_camera_pack_uses_required_cameras() -> None:
    raw = {
        "rgb": {
            "zed_link_camera_0": _rgb(),
            "left_realsense_link_camera_0": _rgb(),
        },
        "depth": {
            "zed_link_camera_0": _depth(),
            "left_realsense_link_camera_0": _depth(),
        },
        "proprio": np.arange(23, dtype=np.float32),
        "language": "navigate",
    }
    packed = pack_obs(
        raw,
        rgb_resolution=256,
        camera_names=["zed_link_camera_0", "left_realsense_link_camera_0"],
    )
    assert set(packed["rgb"]) == {
        "zed_link_camera_0",
        "left_realsense_link_camera_0",
    }
    assert set(packed["depth"]) == set(packed["rgb"])
