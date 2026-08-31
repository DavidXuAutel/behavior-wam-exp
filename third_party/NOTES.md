# Third-party assets

## Allowed

- **Wan2.2** generic pretrained DiT + VAE (+ matching text encoder) as backbone init
- **OmniGibson** + BEHAVIOR-1K for simulation eval
- Official **2026-challenge-demos** (LeRobot V3)
- Challenge baselines **π0.5 / GR00T N1.7** only as external evaluation references on the same subset

## Remote (10.239.121.22:31103) — Week 0

Previous node: `10.239.121.21:31126` (same Ceph home; E0 blocked by GPU occupancy).

```bash
export WAN22_MODEL_DIR=/home/a25689/behavior-wam-exp/checkpoints/wan22
export OMNIGIBSON_ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K/OmniGibson
export BEHAVIOR_DEMOS=/home/a25689/BEHAVIOR-2026/datasets/2026-challenge-demos
```

Wan2.2 TI2V-5B weights are present under `$WAN22_MODEL_DIR` (symlink to an existing
local weight tree). This project does **not** import FastWAM code; only the
generic Wan2.2 checkpoint files are reused.

Frozen values live in `configs/model_lock.yaml` (`status: frozen`).
