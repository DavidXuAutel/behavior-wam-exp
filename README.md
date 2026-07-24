# behavior-wam-exp

Independent MoT World-Action Model for **BEHAVIOR Challenge 2026**.

**Goal:** Skill Planner + MoT-WAM skill executor (Wan2.2 init), legal RGB+Depth+proprio only, optimize **Q-score**. Validate world co-training with A1c vs B. Structure follows leading WAM designs (dual-stream MoT, joint video–action training, FAST deploy).

## Hard boundaries

- Evaluation inputs: **RGB + Depth + proprioception** only (Depth required in the obs pack).
- No BDDL predicates or Q-score as online policy inputs.
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
├── configs/           # model_lock, stages, skills, registry, eval
├── docs/              # pass criteria, specs, plans
├── src/wam/           # MoT video/action experts
├── scripts/
├── wrappers/          # OmniGibson + hierarchical policy + filter
├── reports/
├── tests/
└── third_party/       # Wan2.2 / OmniGibson install notes
```

## Design

See `docs/superpowers/specs/2026-07-21-independent-wam-behavior-design.md` (rev2).

## Status

- [x] Scaffold + independent MoT-WAM design spec (rev2)
- [ ] Task 2: OmniGibson eval smoke + `configs/model_lock.yaml`
- [ ] Task 3: skill episode slicing (nav vs manip)
- [ ] Task 4: Hier-v0 vs Flat
- [ ] Task 5: Depth + candidate_filter
- [ ] Task 6: Stage A0 executor
- [ ] Task 7: A1c vs B
- [ ] Task 8: Stage C + submission pack
