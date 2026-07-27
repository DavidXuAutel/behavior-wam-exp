#!/usr/bin/env python3
"""Validate and optionally freeze configs/model_lock.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "model_lock.yaml"

REQUIRED_TOP = ("status", "wan", "sensors", "temporal", "control", "mot")
REQUIRED_WAN = ("dit_checkpoint", "vae_checkpoint", "text_encoder")
REQUIRED_SENSORS = ("camera_names", "rgb_resolution", "depth_required", "depth_fusion")
REQUIRED_TEMPORAL = (
    "history_frames",
    "action_horizon",
    "execute_steps_per_chunk",
    "denoise_steps_deploy",
)
REQUIRED_CONTROL = ("target_hz", "action_dim", "robot_config")


def load_lock(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"invalid lock file: {path}")
    return data


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return True


def validate(lock: dict) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP:
        if key not in lock:
            errors.append(f"missing top-level key: {key}")

    wan = lock.get("wan") or {}
    for key in REQUIRED_WAN:
        if not _is_filled(wan.get(key)):
            errors.append(f"wan.{key} is null/empty")

    sensors = lock.get("sensors") or {}
    for key in REQUIRED_SENSORS:
        if not _is_filled(sensors.get(key)):
            errors.append(f"sensors.{key} is null/empty")
    if sensors.get("depth_required") is not True:
        errors.append("sensors.depth_required must be true")
    if sensors.get("rgb_resolution") not in (256, 384):
        errors.append("sensors.rgb_resolution must be 256 or 384")

    temporal = lock.get("temporal") or {}
    for key in REQUIRED_TEMPORAL:
        if not _is_filled(temporal.get(key)):
            errors.append(f"temporal.{key} is null/empty")

    control = lock.get("control") or {}
    for key in REQUIRED_CONTROL:
        if not _is_filled(control.get(key)):
            errors.append(f"control.{key} is null/empty")
    if control.get("action_dim") != 23:
        errors.append("control.action_dim must be 23 for R1Pro demos")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=LOCK_PATH)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="exit non-zero unless status == frozen",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="verify local filesystem paths exist (remote machine)",
    )
    args = parser.parse_args()

    lock = load_lock(args.path)
    errors = validate(lock)

    if args.require_frozen and lock.get("status") != "frozen":
        errors.append(f"status is {lock.get('status')!r}, expected 'frozen'")

    if args.check_paths:
        for label, path in (
            ("wan.dit_checkpoint", (lock.get("wan") or {}).get("dit_checkpoint")),
            ("wan.vae_checkpoint", (lock.get("wan") or {}).get("vae_checkpoint")),
            ("wan.text_encoder", (lock.get("wan") or {}).get("text_encoder")),
            (
                "control.robot_config_path",
                (lock.get("control") or {}).get("robot_config_path"),
            ),
        ):
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                errors.append(f"missing path for {label}: {path}")

    if errors:
        print("model_lock INVALID:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"model_lock OK: status={lock.get('status')}")
    print(f"  wan.dit={lock['wan']['dit_checkpoint']}")
    print(f"  cameras={lock['sensors']['camera_names']}")
    print(f"  rgb={lock['sensors']['rgb_resolution']} action_dim={lock['control']['action_dim']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
