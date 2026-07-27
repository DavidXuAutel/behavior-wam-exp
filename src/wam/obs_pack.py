"""Legal observation packing for BEHAVIOR MoT-WAM.

Only RGB + Depth + proprio + language are allowed online.
Depth is required; BDDL / Q-score must never enter this pack.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class MissingDepthError(ValueError):
    """Raised when depth is absent from a raw observation dict."""


def _as_hwc_rgb(arr: np.ndarray, resolution: int | None) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 3 or out.shape[-1] not in (3, 4):
        raise ValueError(f"rgb must be HxWx3/4, got {out.shape}")
    if out.shape[-1] == 4:
        out = out[..., :3]
    if resolution is not None and (out.shape[0] != resolution or out.shape[1] != resolution):
        # nearest resize without hard dependency on cv2/torch
        ys = (np.linspace(0, out.shape[0] - 1, resolution)).astype(np.int64)
        xs = (np.linspace(0, out.shape[1] - 1, resolution)).astype(np.int64)
        out = out[ys][:, xs]
    return out.astype(np.uint8, copy=False)


def _as_hw_depth(arr: np.ndarray, resolution: int | None) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim != 2:
        raise ValueError(f"depth must be HxW, got {out.shape}")
    if resolution is not None and (out.shape[0] != resolution or out.shape[1] != resolution):
        ys = (np.linspace(0, out.shape[0] - 1, resolution)).astype(np.int64)
        xs = (np.linspace(0, out.shape[1] - 1, resolution)).astype(np.int64)
        out = out[ys][:, xs]
    return out.astype(np.float32, copy=False)


def _select_cameras(
    value: Any,
    camera_names: Sequence[str] | None,
    kind: str,
) -> Any:
    if not isinstance(value, Mapping):
        return value
    if not camera_names:
        return dict(value)
    missing = [name for name in camera_names if name not in value]
    if missing:
        raise KeyError(f"missing {kind} cameras: {missing}")
    return {name: value[name] for name in camera_names}


def pack_obs(
    raw: Mapping[str, Any],
    *,
    rgb_resolution: int | None = 256,
    camera_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pack a legal observation dict.

    Required keys in ``raw``: ``rgb``, ``depth``, ``proprio``, ``language``.
    """
    if "depth" not in raw or raw["depth"] is None:
        raise MissingDepthError("depth is required in legal obs pack")

    rgb = _select_cameras(raw["rgb"], camera_names, "rgb")
    depth = _select_cameras(raw["depth"], camera_names, "depth")

    if isinstance(rgb, Mapping):
        packed_rgb = {
            name: _as_hwc_rgb(arr, rgb_resolution) for name, arr in rgb.items()
        }
    else:
        packed_rgb = _as_hwc_rgb(rgb, rgb_resolution)

    if isinstance(depth, Mapping):
        packed_depth = {
            name: _as_hw_depth(arr, rgb_resolution) for name, arr in depth.items()
        }
    else:
        packed_depth = _as_hw_depth(depth, rgb_resolution)

    proprio = np.asarray(raw["proprio"], dtype=np.float32).reshape(-1)
    language = raw.get("language", "")
    if not isinstance(language, str):
        language = str(language)
    if not language.strip():
        raise ValueError("language must be a non-empty string")

    return {
        "rgb": packed_rgb,
        "depth": packed_depth,
        "proprio": proprio,
        "language": language,
    }
