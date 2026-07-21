# Independent WAM for BEHAVIOR 2026 — Design Spec

> Date: 2026-07-21  
> Status: Draft for user review  
> Repo: `behavior-wam-exp`  
> Supersedes: any plan that treated FastWAM / τ₀-WM as a dependency

## 1. Goal

Build a **self-contained World-Action Model (WAM)** for the 2026 BEHAVIOR Challenge:

- Optimize mean **Q-score** (BDDL partial credit) under legal observations only.
- Keep all model code, training, inference, and adapters inside `behavior-wam-exp`.
- Use **generic pretrained Wan2.2** as a frozen/LoRA visual backbone — **not** FastWAM code, configs, checkpoints, or utilities.
- Validate that world/video co-training helps held-out scenes via an A1c vs B ablation.

## 2. Non-goals

- No FastWAM / τ₀-WM imports, weight loading, or copied training entrypoints.
- No privileged simulator state at evaluation time.
- No full-resolution future video generation at test time (too slow for house-scale control).
- No internet-scale video pretraining in this phase (Wan2.2 weights only as init).
- Value head is out of the A0 / A1c / B critical path.

## 3. Constraints (challenge + project)

| Constraint | Value |
|------------|--------|
| Observations | RGB + Depth + proprioception only |
| Embodiment | Default R1Pro (OmniGibson); custom robot config allowed if legal |
| Data | Official BEHAVIOR 2026 demos (LeRobot V3) + skill/language annotations |
| Metric | Primary: mean Q-score; Secondary: full-task success |
| Submission | 100×10×1 rollouts; no cherry-picking |
| Backbone init | Wan2.2 (generic pretrained), managed by this repo |
| FastWAM | Forbidden entirely |

## 4. Architecture

```text
                    ┌─────────────────────┐
  task BDDL + lang  │   Skill Planner     │  → skill_id / skill text
                    └─────────┬───────────┘
                              │
 RGB, Depth, proprio, skill ──┤
                              ▼
                    ┌─────────────────────┐
                    │  Independent WAM    │
                    │  Executor (Fast)    │
                    └─────────┬───────────┘
                              │ action chunk
                              ▼
                    ┌─────────────────────┐
                    │ candidate_filter +  │
                    │ R1Pro env adapter   │
                    └─────────────────────┘
```

### 4.1 Perception / backbone

- Inputs: multi-view RGB, Depth (metric or log-dequantized), proprio, language (skill + task).
- Encode RGB (+ optional depth channel fusion) with **Wan2.2** VAE / DiT backbone.
- Training modes:
  - **FULL**: current + noisy future latents (world loss on future region).
  - **FAST**: current frames only (deployment path).
- Backbone: freeze early; LoRA on later blocks for A1c/B/C.
- Depth: dedicated encoder or concatenated channel pathway; used in filter for near-obstacle penalties.

### 4.2 World tokens

- Hook selected backbone layers → **Perceiver Resampler** → compact world tokens.
- Default: 64 latents, view/time/layer embeddings preserved.
- Mean-pool is an optional ablation only; not the main path.

### 4.3 Action expert

- Independent diffusion / flow-matching action head (owned by this repo).
- Conditions on world tokens + proprio + skill embedding + language.
- Outputs R1Pro action chunks (exact `action_dim` from OmniGibson `r1pro` controller config).
- Deploy: small denoise steps (target 5) + execute first K steps of chunk, then replan.

### 4.4 Skill planner

- Supervised from official skill/subtask annotations.
- Inputs: current obs summary (or frozen visual embeds), BDDL goal text, progress so far.
- Outputs: next skill label (from the 31-skill set) and optional language fragment.
- v0 may be retrieval / VLM-assisted; v1 is a small learned classifier/seq model trained in-repo.
- Skill completion: timeout, subgoal Q-score delta, or annotation-trained terminator.

### 4.5 Safety / control

- `candidate_filter` implemented **in this repo**: reject extreme EE jumps, invalid gripper, depth-near collisions when available.
- Optional multi-seed only for high-risk skills (insert/pour); default single seed for latency.

## 5. Training protocol

Aligned with Apex-WAM-Mini v3.2 *ideas* (fair A1c/B), but **reimplemented here**.

| Stage | What trains | Loss | Purpose |
|-------|-------------|------|---------|
| A0 | Action expert + adapters; backbone frozen; FAST-only | `L_action` | Skill executor baseline |
| A1c | LoRA + adapters + action; FULL/FAST mix; **no** world loss | `L_action` | Compute-matched control |
| B | Same as A1c + world/video loss on FULL | `L_action + λ_w L_world` | WAM hypothesis |
| C | Target fine-tune; mostly FAST; light world regularizer | `L_action + 0.1 L_world` (FULL only) | Deploy |

Locked-same for A1c vs B: identical data mix, LoRA rank, steps, LR, forward mix — **only** `λ_w` differs (0 vs 0.3).

Additional optional term after Hier-v0 works: `L_skill_progress` from BDDL predicate progress labels derived offline from demos/sim labels **for training only** (never used as eval privileged input).

## 6. Data flow

1. Mount official `2026-challenge-demos` (LeRobot V3).
2. Slice episodes by skill annotations → skill-level windows.
3. `data_compatibility.yaml` registry: only verified R1Pro BEHAVIOR demos get `action` supervision; others video-only if any.
4. Minimum viable Week 0–1: 5–10 tasks, not full 3TB.
5. Failure rollouts collected later for Stage C only; disjoint from eval instances.

## 7. Evaluation

- OmniGibson challenge evaluator; legal obs only.
- Early gate: public subset (`configs/eval_public10.yaml`).
- Contrasts:
  1. Flat WAM (no planner) vs Hier-v0
  2. A1c vs B on held-out / new scenes
- Pass criteria: see `docs/pass_criteria.yaml`
  - Hier vs Flat: +0.05 Q-score (macro), CI lower bound > 0
  - B vs A1c: +0.03 Q-score on held-out scenes, CI lower bound > 0
  - Per-task drop vs A1c ≤ 0.05

## 8. Repo layout (target)

```text
behavior-wam-exp/
├── configs/                 # stages, skills, data registry, eval
├── docs/
│   ├── pass_criteria.yaml
│   ├── plans/
│   └── superpowers/specs/   # this design
├── src/wam/                 # independent model code (backbone hooks, perceiver, action expert)
├── wrappers/                # OmniGibson + hierarchical policy + filter
├── scripts/                 # smoke, slice, train, eval, pack
├── tests/
├── reports/
└── third_party/             # Wan2.2 / OmniGibson install notes only (no FastWAM)
```

## 9. Interfaces (stable contracts)

```python
# Planner
def next_skill(obs: dict, task: dict) -> str: ...

# Executor (WAM Fast path)
def act(obs: dict, skill: str) -> np.ndarray:  # shape [H, action_dim] or flat env action

# Eval wrapper
class BehaviorEvalWrapper:
    def reset(self, task_info: dict) -> None: ...
    def act(self, obs: dict) -> np.ndarray: ...
```

Observation dict keys (legal): `rgb`, `depth`, `proprio`, plus language fields from task metadata. No segmentation / object poses / global pose.

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Long-horizon compounding error | Hierarchical skills + replan on subgoal stall |
| Wan2.2 + multi-view latency | FAST path; chunked actions; single seed default |
| Action-space mismatch | Freeze `action_dim` from `r1pro.yaml` before any train |
| A1c/B unfairness | Locked-same table + `backbone_forward_count` log |
| Scope creep | No FastWAM; no value head on critical path; no 14B until B passes |

## 11. Approval record

- Architecture choice (2026-07-21): Independent WAM + generic Wan2.2 init; no FastWAM.
- User confirmation: architecture confirmed.

## 12. Next step after this spec is approved

Rewrite the shortest implementation plan under `docs/plans/` with FastWAM removed, then execute Task 2+ inside this repo only.
