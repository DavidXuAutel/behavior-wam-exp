# behavior-wam-exp

Independent World-Action Model for **BEHAVIOR Challenge 2026**.

**Goal:** Skill Planner + independent WAM skill executor, legal observations only, optimize **Q-score**. Validate world co-training with A1c vs B. Backbone init: generic **Wan2.2** (not FastWAM).

## Hard boundaries

- **No FastWAM / τ₀-WM**: no code, configs, checkpoints, or utilities from those repos.
- Evaluation inputs: **RGB + Depth + proprioception** only.
- Primary metric: mean **Q-score**. Full-task success is secondary.
- Submissions must be reproducible; no cherry-picking rollouts.

## Challenge links

- Challenge: https://behavior.stanford.edu/challenge/
- Evaluation & Rules: https://behavior.stanford.edu/challenge/evaluation.html
- Dataset: https://behavior.stanford.edu/challenge/dataset.html
- Submission: https://behavior.stanford.edu/challenge/submission.html

## Layout

```text
behavior-wam-exp/
├── configs/           # stages, skills, data compatibility, eval subsets
├── docs/              # pass criteria, specs, plans
├── src/wam/           # independent WAM implementation
├── scripts/           # smoke / data / train / eval / pack
├── wrappers/          # OmniGibson eval + hierarchical policy
├── reports/
├── tests/
└── third_party/       # Wan2.2 / OmniGibson install notes only
```

## Design

See `docs/superpowers/specs/2026-07-21-independent-wam-behavior-design.md`.

## Status

- [x] Scaffold + independent-WAM design spec
- [ ] Task 2: OmniGibson eval smoke
- [ ] Task 3: skill episode slicing
- [ ] Task 4: Hier-v0 vs Flat
- [ ] Task 5: Depth + candidate_filter
- [ ] Task 6: Stage A0 executor
- [ ] Task 7: A1c vs B
- [ ] Task 8: Stage C + submission pack
