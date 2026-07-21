# Third-party dependencies (allowed)

This experiment owns its WAM implementation. External assets are limited to:

## Allowed

- **Wan2.2** generic pretrained weights (Hugging Face / DiffSynth) as backbone init
- **OmniGibson** + BEHAVIOR-1K assets for sim eval
- Official **2026-challenge-demos** (LeRobot V3)

## Forbidden

- FastWAM repository code, configs, checkpoints, scripts
- τ₀-WM / TauPolicy as a dependency
- Any “read-only mount” of FastWAM into training or inference

## Suggested env

```bash
export WAN22_MODEL_DIR="$HOME/checkpoints/wan22"   # or HF cache path
export OMNIGIBSON_ROOT="$HOME/Projects/OmniGibson" # adjust
export BEHAVIOR_DEMOS="$HOME/data/2026-challenge-demos"
```

Document exact download commands in Task 2 notes when installed.
