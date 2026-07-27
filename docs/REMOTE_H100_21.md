# Remote machine: 10.239.121.21 (2×H100)

## Status (2026-07-27)

| Item | Value |
|------|--------|
| SSH | `ssh a25689@10.239.121.21 -p 31126` |
| GPUs | 2× NVIDIA H100 80GB HBM3 (idle at probe) |
| Home storage | Ceph ~157T, ~87T free |
| Official demos | `/home/a25689/BEHAVIOR-2026/datasets/2026-challenge-demos` (~3.0T) |
| Episodes / tasks | 20000 / 100 (LeRobot v3.0, R1Pro, action_dim **23**) |
| Annotations | 100 `task-*` dirs + skill summaries |
| Raw HDF5 | **not present** (optional for D2 perturb replay) |
| Code workspace | `/home/a25689/behavior-wam-exp` |
| BEHAVIOR-1K | `/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K` |
| Week0 report | `behavior-wam-exp/reports/week0_env.json` |
| model_lock | `draft` — Wan weights still TODO |

## Tier implication

2×H100 = **MVP / Base-lite**. A1c+B wall-clock ≈ **2–4×** slower than 8-GPU Base estimate. Prefer:

- RGB **256**, start **head (+ optional 1 wrist)** not full 3-view
- `coupling: perceiver_cross_attn` for A0
- Parallelize A1c ∥ B only if a second 2-GPU node appears

## Next commands (on remote)

```bash
# 1) Watch env bootstrap (started in background if launched)
tail -f ~/behavior-wam-exp/reports/bootstrap_env.log

# 2) After env OK:
eval "$(/home/a25689/bin/micromamba shell hook -s bash)"
micromamba activate mot-wam

# 3) Download Wan2.2 5B weights into ~/behavior-wam-exp/checkpoints/wan22
#    then fill configs/model_lock.yaml wan.* paths and freeze status

# 4) D0 data audit (5–10 task subset first)
# 5) OmniGibson E0 smoke via BEHAVIOR-1K
```

## Do not

- Overwrite `kairos` conda env
- Mix FastWAM tree into this training path
- Start A0 before `model_lock.status: frozen` and `reports/data_audit.json` pass
