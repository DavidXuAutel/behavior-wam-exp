#!/usr/bin/env python3
import datetime
import json
from pathlib import Path

p = Path("/home/a25689/behavior-wam-exp/reports/week0_env.json")
d = json.loads(p.read_text())
d["timestamp_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
d.setdefault("torch_probe", {})["mot-wam"] = "2.6.0+cu126 12.6 2"
d["model_lock"] = {
    "status": "frozen",
    "path": "/home/a25689/behavior-wam-exp/configs/model_lock.yaml",
    "rgb_resolution": 256,
    "action_dim": 23,
}
d["wan22"] = {
    "path": "/home/a25689/behavior-wam-exp/checkpoints/wan22",
    "verified": True,
    "size": "~32G",
}
d["data_audit"] = {
    "status": "pass",
    "subset_tasks": 8,
    "action_files_spotchecked": 100,
    "action_dim": 23,
}
d["status"] = "week0_ready"
p.write_text(json.dumps(d, indent=2) + "\n")
print(d["status"])
