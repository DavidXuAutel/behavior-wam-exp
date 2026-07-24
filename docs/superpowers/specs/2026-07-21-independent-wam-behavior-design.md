# Independent WAM for BEHAVIOR 2026 — Design Spec

> Date: 2026-07-24 (rev2)  
> Status: Approved architecture + P0 fixes incorporated  
> Repo: `behavior-wam-exp`

## 1. Goal

Build a **self-contained World-Action Model (WAM)** for the 2026 BEHAVIOR Challenge:

- Optimize mean **Q-score** (BDDL partial credit) under legal observations only.
- Keep all model code, training, inference, and adapters inside `behavior-wam-exp`.
- Initialize from **generic Wan2.2** video DiT weights; all robot-facing modules are trained here.
- Adopt structural lessons from current top WAM systems (LingBot-VA, Motus, and related MoT video-action models): **dual-stream MoT**, **joint video–action training**, **causal chunk + observation feedback**, **inference without future-video denoising**.
- Validate that world co-training helps held-out scenes via an A1c vs B ablation.

## 2. Non-goals

- No privileged simulator state at evaluation time (no BDDL/Q-score as online policy input).
- No full-resolution future video generation at test time (too slow for house-scale control).
- No internet-scale video pretraining in this phase (Wan2.2 weights only as init).
- No Progress/Value expert on the A0 / A1c / B critical path (optional after Stage C only).
- No 14B backbone or DINO dual-track until B passes on the public subset.

## 3. Constraints

| Constraint | Value |
|------------|--------|
| Observations | RGB + Depth + proprioception only |
| Embodiment | Default R1Pro (OmniGibson); custom robot config allowed if legal |
| Data | Official BEHAVIOR 2026 demos (LeRobot V3) + skill/language annotations |
| Metric | Primary: mean Q-score; Secondary: full-task success |
| Submission | 100×10×1 rollouts; no cherry-picking |
| Backbone init | Wan2.2-TI2V / Wan2.2 5B-class DiT (exact id locked in §4.1) |

## 4. Architecture

```text
 task BDDL text + language
            │
            ▼
   ┌────────────────────┐
   │  Skill Planner v0  │──► skill_id ∈ {nav, manip skills}
   └─────────┬──────────┘
             │
 RGB+Depth+proprio+skill ──► Independent MoT-WAM Executor (deploy: FAST)
             │
             ▼
   ┌────────────────────┐
   │ mode router        │──► NavigationController | ManipulationController
   │ candidate_filter   │
   │ R1Pro action pack  │──► env action (base + torso + arms + grippers)
   └────────────────────┘
```

### 4.1 Perception / backbone (locked engineering contract)

Inputs always packed together (Depth is **required**, not optional):

- Multi-view **RGB**
- Multi-view **Depth** (metric after dequantization; also feeds near-obstacle filter)
- **Proprio** (joints / EE / gripper / base as exposed by R1Pro)
- **Language**: task instruction + current skill text

| Item | Lock (Week 0 must freeze before A0) |
|------|--------------------------------------|
| Checkpoint family | Wan2.2 5B-class video DiT + matching Wan VAE |
| Text encoder | Frozen UMT5 / T5-XXL compatible with chosen Wan stack |
| Cameras (default R1Pro) | Head + left wrist + right wrist (exact names from robot yaml) |
| RGB resolution | Start 256² or 384² per view (benchmark both; pick one before A1c/B) |
| Depth | Same spatial grid as RGB; separate stem or 4th-channel fusion — choose in Week 0 smoke |
| Temporal context | Current keyframe (+ short history ≤ 4 frames if VRAM allows) |
| Deploy control target | ≥ 3–5 Hz closed-loop on eval GPU class |
| Train modes | FULL (current + noisy future latents) / FAST (current only) |
| Backbone trainability | Freeze early blocks; LoRA on later blocks for A1c/B/C |

Record the chosen values in `configs/model_lock.yaml` before any Stage A0 run.

### 4.2 MoT-WAM core (aligned with top WAM designs)

Reference pattern from leading WAMs (LingBot-VA MoT dual-stream; Motus unified latent experts; train-time world modeling without test-time future-video rollout):

```text
Wan VAE (frozen) ──► video latent tokens
UMT5 (frozen)     ──► text tokens
Depth stem        ──► depth tokens (fused or cross-attn into video stream)
State MLP         ──► proprio tokens
Skill embed       ──► skill tokens

MoT DiT (shared layers, modality-specific experts):
  Video Expert  — world / dynamics stream (inherits Wan init + LoRA)
  Action Expert — flow-matching / diffusion action chunk head (trained from scratch or light init)

Shared self-attention across streams with structured / block-causal masks.
```

**Training (FULL):** joint flow-matching on video future latents + action chunks  
`L = L_action + λ_w L_world` (B); A1c uses the same graph with `λ_w = 0`.

**Deployment (FAST):** single forward of Video Expert on **clean current** latents only → Action Expert denoises actions (≈5 steps). **No** iterative future-video denoising at test time.

**Closed loop:** execute first `K` actions of the chunk; ingest new real observations; replan. Optional short KV cache of recent observation tokens for temporal consistency (causal chunk feedback), without generating imagined frames.

Default Feature path inside Video Expert: last layers’ tokens → compact context for Action Expert (Perceiver resampler, 64 latents). Mean-pool is ablation-only.

### 4.3 Action expert & whole-body control

- Outputs R1Pro action chunks with `action_dim` taken **verbatim** from OmniGibson `r1pro` controller config before training.
- Split dimensions conceptually:
  - **Base / navigation** (planar twist or joint base cmds)
  - **Torso**
  - **Bimanual arms + grippers**
- Mode router (from planner skill type):
  - `navigate_*` / `move to` → emphasize base cmds; clamp large arm deltas
  - manipulation skills → emphasize arms/grippers; limit base unless skill needs approach

### 4.4 Skill planner (frozen v0 definition)

**Planner v0 (required for Hier gate):** annotation-trained **skill classifier / next-skill head** over the official 31-skill set (+ explicit `navigate` grouping if annotations collapse move-to). Inputs: frozen visual embeds (or cached backbone features) + task language + last skill. **No VLM-in-the-loop for the gate experiment.**

**Planner v1 (optional later):** VLM / retrieval assistant; only after Hier-v0 gate passes.

**Skill completion (eval-legal only):**

| Allowed online | Forbidden online |
|----------------|------------------|
| Timeout / max steps per skill | Reading BDDL goal predicates |
| Learned terminator head (trained offline) | Reading Q-score / evaluator metrics |
| Vision+proprio heuristics (e.g. gripper open, base speed≈0) | Any privileged sim state |

Offline, demos may be labeled with BDDL progress for **training** `L_skill_progress` or terminator labels — never fed as policy observations at eval.

### 4.5 Navigation vs manipulation

| Skill class | Examples | Control emphasis | Completion heuristic (examples) |
|-------------|---------|------------------|----------------------------------|
| Navigation | move to, push to (approach) | Base velocity / path following; depth stop | Near-goal visual cue or timeout |
| Manipulation | pick, place, open, pour, wipe… | Arms + grippers; base mostly held | Terminator / gripper+contact heuristics |

Depth near-field is **mandatory** in `candidate_filter` for both classes (collision / cliff-like stop).

### 4.6 Safety / control

Implemented in-repo:

- Reject extreme EE jumps, invalid gripper range, NaNs
- Depth proximity penalty / hard reject below threshold
- Optional 2–3 seed sampling only for high-risk skills (insert, pour, attach); default single seed

## 5. Training protocol

| Stage | What trains | Loss | Purpose |
|-------|-------------|------|---------|
| A0 | Action expert + stems; Video Expert frozen; FAST-only | `L_action` | Skill executor baseline |
| A1c | Video LoRA + Action; FULL/FAST mix; **no** world loss | `L_action` | Compute-matched control |
| B | Same as A1c + world loss on FULL | `L_action + λ_w L_world` | WAM hypothesis |
| C | Target fine-tune; mostly FAST; light world regularizer | `L_action + 0.1 L_world` on FULL only | Deploy |

Locked-same for A1c vs B: identical data mix, LoRA rank, steps, LR, forward mix, MoT depth — **only** `λ_w` differs (0 vs 0.3). Log `backbone_forward_count` and GPU-hours.

Optional after Hier-v0: `L_skill_progress` from offline BDDL-derived labels.

## 6. Data flow

1. Mount official `2026-challenge-demos` (LeRobot V3).
2. Slice episodes by skill annotations → skill-level windows; tag `nav` vs `manip`.
3. `configs/data_compatibility.yaml`: only verified R1Pro BEHAVIOR demos get action supervision.
4. Week 0–1 minimum: 5–10 tasks (not full corpus).
5. Failure rollouts for Stage C only; disjoint from eval instances.

## 7. Evaluation

- OmniGibson challenge evaluator; legal obs only.
- Early gate: `configs/eval_public10.yaml`.
- Contrasts (same subset, same seeds protocol):
  1. **Flat MoT-WAM** (task language only, no skill planner) vs **Hier-v0**
  2. **A1c vs B** on held-out / new scenes
  3. **External baseline row**: official π0.5 and/or GR00T N1.7 on the same subset (report only; not a training dependency)
- Pass criteria: `docs/pass_criteria.yaml`
  - Hier vs Flat: +0.05 Q-score (macro), bootstrap CI lower bound > 0
  - B vs A1c: +0.03 Q-score on held-out scenes, CI lower bound > 0
  - Per-task drop vs A1c ≤ 0.05
- Before claiming pass/fail: run a **power check** (subset size × trials) so +0.03/+0.05 is detectable.

## 8. Repo layout

```text
behavior-wam-exp/
├── configs/                 # model_lock, stages, skills, registry, eval
├── docs/
│   ├── pass_criteria.yaml
│   ├── plans/
│   └── superpowers/specs/
├── src/wam/                 # MoT video/action experts, stems, losses
├── wrappers/                # OmniGibson, hierarchical policy, filter, nav/manip router
├── scripts/
├── tests/
├── reports/
└── third_party/             # Wan2.2 + OmniGibson install notes
```

## 9. Interfaces

```python
def next_skill(obs: dict, task: dict) -> str: ...

def act(obs: dict, skill: str) -> np.ndarray:
    # env-ready action or chunk flattened per robot.action_dim

class BehaviorEvalWrapper:
    def reset(self, task_info: dict) -> None: ...
    def act(self, obs: dict) -> np.ndarray: ...
```

Legal obs keys: `rgb`, `depth`, `proprio`, plus language from task metadata. No segmentation, object poses, global pose, BDDL predicates, or evaluator scores.

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Long-horizon compounding | Skill hierarchy + legal completion + replan |
| House-scale latency | FAST deploy; no future-video denoising; single seed default |
| Action-space mismatch | Freeze `action_dim` from `r1pro.yaml` before train |
| Nav failures dominate Q-score | Explicit nav skills + depth stop + base-first routing |
| A1c/B unfairness | Locked-same + forward_count log |
| Trailing SOTA VLAs on leaderboard | Always report π0.5/GR00T on same subset; decide distillation/fusion only after B gate |
| Scope creep | No 14B / DINO / Value expert until B passes |

## 11. Why this structure (SOTA WAM lessons)

| Lesson | Source family | Adopted here |
|--------|---------------|--------------|
| Dual-stream MoT, shared attention | LingBot-VA, Motus | Video Expert + Action Expert |
| Joint video–action generative training | LingBot-VA, Motus, DreamZero-class WAMs | FULL `L_world + L_action` |
| Causal chunk + real observation feedback | LingBot-VA | Closed-loop replan / optional KV of real frames |
| World modeling value is mostly from **training** signal | Controlled WAM ablations showing train video ≫ test imagination | Deploy FAST; keep A1c vs B |
| House-scale needs Hz, not pixel futures | Challenge constraint | No test-time future-video denoising |

## 12. Approval / revision record

- 2026-07-21: Independent Wan2.2-init WAM architecture confirmed.
- 2026-07-24: P0 fixes — legal skill completion, nav/manip split, model lock table; MoT structure upgraded to match top WAM designs; removed third-party policy-code dependencies from the narrative.

## 13. Next step

Rewrite the shortest implementation plan under `docs/plans/` to match this rev2 spec, then execute Week 0 (`model_lock.yaml` + OmniGibson smoke).
