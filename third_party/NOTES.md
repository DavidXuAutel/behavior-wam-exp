# Third-party assets

## Allowed

- **Wan2.2** generic pretrained DiT + VAE (+ matching text encoder) as backbone init
- **OmniGibson** + BEHAVIOR-1K for simulation eval
- Official **2026-challenge-demos** (LeRobot V3)
- Challenge baselines **π0.5 / GR00T N1.7** only as external evaluation references on the same subset

## Suggested env

```bash
export WAN22_MODEL_DIR="$HOME/checkpoints/wan22"
export OMNIGIBSON_ROOT="$HOME/Projects/OmniGibson"
export BEHAVIOR_DEMOS="$HOME/data/2026-challenge-demos"
```

Exact download commands and the frozen `configs/model_lock.yaml` values are filled in Week 0.
