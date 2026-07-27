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

## Design & plan

- Spec (rev2): `docs/superpowers/specs/2026-07-21-independent-wam-behavior-design.md`
- **Train & eval scheme:** `docs/superpowers/specs/2026-07-27-train-eval-scheme.md`
- **Landable eng. plan:** `docs/superpowers/plans/2026-07-24-independent-mot-wam-landable.md`
- Merged export: `docs/COMPLETE_SCHEME.md`

## Status

- [x] Spec rev2 + landable implementation plan
- [ ] W0: model_lock freeze + OmniGibson 1×1 smoke
- [ ] W1: skill slices + planner v0 + filter/router
- [ ] W2: MoT A0 + Hier vs Flat gate
- [ ] W3: A1c vs B + Stage C submission pack
