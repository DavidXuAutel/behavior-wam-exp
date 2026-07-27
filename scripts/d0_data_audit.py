#!/usr/bin/env python3
"""D0 demo audit for a small task subset (episode-level checks)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


REQUIRED_OBS_PREFIXES = (
    "observation.rgb.",
    "observation.depth_linear.",
    "observation.state",
)


def load_info(demos: Path) -> dict:
    return json.loads((demos / "meta" / "info.json").read_text())


def load_tasks(demos: Path) -> list[str]:
    tasks_path = demos / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = []
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                tasks.append(obj)
            elif isinstance(obj, dict):
                tasks.append(obj.get("task") or obj.get("task_name") or str(obj))
            else:
                tasks.append(str(obj))
        return tasks
    # fallback: annotation dirs
    ann = demos / "annotations"
    return sorted(p.name for p in ann.glob("task-*"))


def sample_parquet_action_dim(demos: Path, n: int, seed: int) -> dict:
    """Best-effort action dim spot-check via pyarrow if available."""
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        return {"status": "skipped", "reason": f"pyarrow unavailable: {e}"}

    data_root = demos / "data"
    files = sorted(data_root.rglob("*.parquet"))
    if not files:
        return {"status": "fail", "reason": "no parquet files under data/"}

    rng = random.Random(seed)
    chosen = files if len(files) <= n else rng.sample(files, n)
    dims = []
    errors = []
    for fp in chosen:
        try:
            pf = pq.ParquetFile(fp)
            schema = pf.schema_arrow
            # LeRobot often stores action as fixed-size list / list
            names = set(schema.names)
            if "action" not in names:
                errors.append(f"{fp}: missing action column")
                continue
            # read one row group tiny slice
            table = pf.read_row_group(0, columns=["action"])
            col = table.column(0)
            # inspect first non-null
            val = col[0].as_py()
            if isinstance(val, (list, tuple)):
                dims.append(len(val))
            else:
                errors.append(f"{fp}: unexpected action type {type(val)}")
        except Exception as e:
            errors.append(f"{fp}: {e}")

    ok = all(d == 23 for d in dims) and not errors
    return {
        "status": "pass" if ok and dims else ("fail" if errors else "fail"),
        "n_files": len(chosen),
        "dims": Counter(dims),
        "errors": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos",
        type=Path,
        default=Path("/home/a25689/BEHAVIOR-2026/datasets/2026-challenge-demos"),
    )
    parser.add_argument("--n-tasks", type=int, default=8)
    parser.add_argument("--n-action-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/a25689/behavior-wam-exp/reports/data_audit.json"),
    )
    args = parser.parse_args()

    demos = args.demos
    info = load_info(demos)
    tasks = load_tasks(demos)
    rng = random.Random(args.seed)
    subset = tasks if len(tasks) <= args.n_tasks else sorted(rng.sample(tasks, args.n_tasks))

    features = info.get("features") or {}
    feature_keys = list(features.keys())
    has_rgb = any(k.startswith("observation.rgb.") for k in feature_keys)
    has_depth = any(k.startswith("observation.depth_linear.") for k in feature_keys)
    has_state = "observation.state" in feature_keys
    action = features.get("action") or {}
    action_shape = action.get("shape")

    cameras = sorted(
        k.split("observation.rgb.", 1)[1]
        for k in feature_keys
        if k.startswith("observation.rgb.")
    )

    ann_root = demos / "annotations"
    ann_ok = []
    ann_missing = []
    for t in subset:
        # annotation dirs are like task-000 or task names
        candidates = [
            ann_root / t,
            ann_root / f"task-{t}",
            ann_root / t.replace("task-", ""),
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            # also accept exact task-* names already
            if (ann_root / t).exists():
                found = ann_root / t
        if found is None:
            ann_missing.append(t)
        else:
            ann_ok.append(str(found))

    action_spot = sample_parquet_action_dim(demos, args.n_action_files, args.seed)

    checks = {
        "info_present": True,
        "total_episodes": info.get("total_episodes"),
        "total_tasks": info.get("total_tasks"),
        "robot_type_r1pro": info.get("robot_type") == "R1Pro",
        "action_shape_23": action_shape == [23] or action_shape == 23,
        "has_rgb": has_rgb,
        "has_depth": has_depth,
        "has_state": has_state,
        "subset_annotations_found": len(ann_missing) == 0,
    }
    passed = all(
        [
            checks["robot_type_r1pro"],
            checks["action_shape_23"],
            checks["has_rgb"],
            checks["has_depth"],
            checks["has_state"],
            checks["subset_annotations_found"],
            action_spot.get("status") in ("pass", "skipped"),
        ]
    )

    report = {
        "status": "pass" if passed else "fail",
        "demos": str(demos),
        "subset_tasks": subset,
        "cameras": cameras,
        "checks": checks,
        "annotation_ok": ann_ok,
        "annotation_missing": ann_missing,
        "action_spotcheck": action_spot,
        "notes": [
            "Legal obs audit is schema-level (RGB+Depth+state present).",
            "Episode-level Q/BDDL never fed to policy; this audit does not load privileged labels into model inputs.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "out": str(args.out)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
