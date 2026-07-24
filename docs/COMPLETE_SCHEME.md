# BEHAVIOR 2026 × Independent MoT-WAM — 完整方案（合并版）

> Repo: `behavior-wam-exp`  
> 合并日期: 2026-07-24  
> 组成: 设计规格 rev2 + 通过线 + model_lock 模板 + 可落地实施计划  
> 权威分册仍以仓库内原文件为准；本文件用于一次性通读/导出。

**分册路径：**

| 分册 | 路径 |
|------|------|
| 设计规格 | `docs/superpowers/specs/2026-07-21-independent-wam-behavior-design.md` |
| 可落地计划 | `docs/superpowers/plans/2026-07-24-independent-mot-wam-landable.md` |
| 通过线 | `docs/pass_criteria.yaml` |
| 模型锁定 | `configs/model_lock.yaml` |

---

# 第一部分：设计规格（rev2）

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

**Navigation capability source:** official demo skill segments (esp. `move to`) supervise the Action Expert’s base dims; Planner selects nav skills; depth stop avoids collisions. No separate SLAM/map stack in this phase.

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
- 2026-07-24: P0 fixes — legal skill completion, nav/manip split, model lock table; MoT structure upgraded to match top WAM designs.

## 13. Next step

Execute landable plan Week 0 (`model_lock.yaml` + OmniGibson smoke).

---

# 第二部分：通过线（pass_criteria.yaml）

```yaml
# Pass criteria (BEHAVIOR × Independent MoT-WAM)

macro:
  primary: q_score_mean
  vs_flat_wam_executor: 0.05
  vs_a1c_on_heldout_scenes: 0.03
  bootstrap_ci_lower_bound_above_zero: true

per_task:
  max_qscore_drop_vs_a1c: 0.05

engineering:
  observation_legal: true
  no_bddl_or_qscore_as_policy_input: true
  depth_required_in_obs_pack: true
  model_lock_frozen_before_a0: true
  submission_reproducible: true

baselines:
  report_pi05_or_groot_on_same_subset: true

checkpoint_selection:
  primary_metric: q_score_or_legal_subgoal_proxy
  secondary_metric: clean_success_rate
  tie_breaker: lower_val_action_loss

power_check:
  target_mde_macro: 0.03
  require_documented_before_gate_claim: true
```

---

# 第三部分：model_lock 模板（configs/model_lock.yaml）

```yaml
# model_lock.yaml — freeze before Stage A0
# Fill during Week 0 smoke; do not start A0 until status: frozen

status: draft  # -> frozen

wan:
  dit_checkpoint: null
  vae_checkpoint: null
  text_encoder: null

sensors:
  camera_names: []
  rgb_resolution: null          # 256 or 384
  depth_required: true
  depth_fusion: null            # separate_stem | channel_concat

temporal:
  history_frames: 1
  action_horizon: null
  execute_steps_per_chunk: null
  denoise_steps_deploy: 5

control:
  target_hz: null
  action_dim: null
  robot_config: r1pro.yaml

mot:
  video_expert: wan22_lora
  action_expert: flow_matching_dit
  shared_attention: true
  perceiver_latents: 64
  deploy_mode: fast
```

---

# 第四部分：可落地实施计划

# Independent MoT-WAM × BEHAVIOR — Landable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a runnable hierarchical MoT-WAM on BEHAVIOR 2026 (legal RGB+Depth+proprio), with Week-gated proofs: smoke → Hier vs Flat → A0 → A1c vs B → Stage C submission pack.

**Architecture:** Skill Planner v0 (annotation classifier) selects nav/manip skills; dual-stream MoT (Wan2.2 Video Expert + Action Expert) executes under FAST deploy; depth-aware candidate_filter + R1Pro adapter; train FULL with optional `L_world` for A1c/B.

**Tech Stack:** Python 3.10+ · PyTorch · Wan2.2 DiT/VAE/UMT5 · OmniGibson / BEHAVIOR-1K · LeRobot V3 demos · pytest · Hydra or plain YAML configs

**Spec:** `docs/superpowers/specs/2026-07-21-independent-wam-behavior-design.md` (rev2)  
**Pass criteria:** `docs/pass_criteria.yaml`

## Global Constraints

- Observations at eval: RGB + Depth + proprio only; Depth **required** in obs pack.
- Never feed BDDL predicates or Q-score into the online policy.
- Primary metric: mean Q-score; no cherry-picking rollouts.
- Freeze `configs/model_lock.yaml` (`status: frozen`) before Stage A0.
- Navigation = demo skill imitation + planner nav skills + depth stop — no separate map stack in this plan.
- No 14B / DINO / Value expert until B gate passes.
- All code lives in `behavior-wam-exp`.

## File map (target)

```text
configs/
  model_lock.yaml
  skill_set.yaml
  data_compatibility.yaml
  eval_public10.yaml
  stage_a0.yaml | stage_a1c.yaml | stage_b.yaml | stage_c.yaml
src/wam/
  obs_pack.py
  encoders.py
  mot.py
  losses.py
  policy.py
  planner_v0.py
src/data/
  lerobot_behavior.py
  skill_slice.py
wrappers/
  behavior_eval_wrapper.py
  hierarchical_policy.py
  r1pro_action_adapter.py
  candidate_filter.py
  mode_router.py
scripts/
  smoke_omnigibson_eval.sh
  freeze_model_lock.py
  build_skill_episodes.py
  train_planner_v0.py
  train_stage.py
  eval_subset.py
  report_bootstrap.py
  pack_submission.sh
tests/
  test_obs_pack.py
  test_skill_class.py
  test_action_adapter.py
  test_candidate_filter.py
  test_mot_shapes.py
reports/
```

---

### Task 0: Week-0 environment + `model_lock` freeze

**Files:**
- Modify: `configs/model_lock.yaml`
- Create: `scripts/freeze_model_lock.py`
- Modify: `third_party/NOTES.md`
- Create: `reports/week0_env.json`

**Interfaces:**
- Produces: frozen fields `wan.*`, `sensors.*`, `temporal.*`, `control.action_dim`, `control.target_hz`

- [ ] Install OmniGibson + BEHAVIOR assets; record paths in `reports/week0_env.json`
- [ ] Download Wan2.2 5B-class DiT + VAE + text encoder; write paths into `model_lock.yaml`
- [ ] Load default `r1pro.yaml`; set `control.action_dim` and `sensors.camera_names`
- [ ] Pick `rgb_resolution` ∈ {256, 384} after one forward VRAM probe; set `depth_fusion`
- [ ] Set `status: frozen` only when all nulls filled
- [ ] Commit: `chore: freeze model_lock for BEHAVIOR MoT-WAM`

**Gate:** `model_lock.status == frozen` and `reports/week0_env.json` exists.

---

### Task 1: Legal obs pack + unit tests

**Files:**
- Create: `src/wam/obs_pack.py`
- Test: `tests/test_obs_pack.py`

**Interfaces:**
- Produces: `pack_obs(raw: dict) -> dict` with keys `rgb`, `depth`, `proprio`, `language`; raises if depth missing

- [ ] Write failing tests: missing depth raises; packed shapes match lock resolution
- [ ] Implement `pack_obs`
- [ ] `pytest tests/test_obs_pack.py -v` PASS
- [ ] Commit: `feat: legal obs pack requiring depth`

**Gate:** tests green.

---

### Task 2: OmniGibson smoke (1 task × 1 instance)

**Files:**
- Modify: `scripts/smoke_omnigibson_eval.sh`
- Modify: `wrappers/behavior_eval_wrapper.py`
- Create: `reports/smoke_eval.json`

**Interfaces:**
- Consumes: `pack_obs`, random or zero action policy
- Produces: one official metrics JSON path listed in `reports/smoke_eval.json`

- [ ] Wire official evaluator command (document full CLI in smoke script)
- [ ] Run dummy policy through wrapper for 1×1 rollout
- [ ] Assert metrics file exists; assert obs path never requests privileged keys
- [ ] Commit: `feat: OmniGibson 1x1 smoke eval`

**Gate:** `reports/smoke_eval.json` with `"status": "ok"` and metrics path.

---

### Task 3: Skill slicing (nav vs manip) — minimal 5–10 tasks

**Files:**
- Create: `src/data/lerobot_behavior.py`
- Create: `src/data/skill_slice.py`
- Create: `scripts/build_skill_episodes.py`
- Modify: `configs/data_compatibility.yaml` (`verified` after checks)
- Create: `reports/registry_verification/behavior_demos.json`

**Interfaces:**
- Produces: on-disk skill windows `{obs, actions, skill, skill_class, language}` under `data/skill_episodes/` (gitignored)

- [ ] Reader for LeRobot V3 demos + annotations
- [ ] Slice by skill; tag `navigation` vs `manipulation` using `configs/skill_set.yaml` + `move to` rules
- [ ] Build **min subset**: 5–10 tasks only
- [ ] Verify 100 random windows: action_dim match, gripper range, skill label non-empty → write registry JSON; set `verified: true`
- [ ] Commit: `feat: skill episode slicer for BEHAVIOR demos`

**Gate:** `verified: true` and ≥1k skill windows on disk (or documented count in report).

---

### Task 4: R1Pro adapter + candidate_filter + mode router

**Files:**
- Modify: `wrappers/r1pro_action_adapter.py`
- Create: `wrappers/candidate_filter.py`
- Create: `wrappers/mode_router.py`
- Modify: `wrappers/hierarchical_policy.py`
- Test: `tests/test_action_adapter.py`, `tests/test_candidate_filter.py`, `tests/test_skill_class.py`

**Interfaces:**
- `R1ProActionAdapter.to_env(action) -> np.ndarray` shape `[action_dim]`
- `candidate_filter(action, obs) -> np.ndarray` (may clamp/reject→safe hold)
- `mode_router(skill, action_chunk) -> action_chunk` (nav: damp arms; manip: damp base)

- [ ] Failing tests for dim mismatch, gripper OOB, depth-near hard reject
- [ ] Implement adapter/filter/router
- [ ] pytest PASS
- [ ] Commit: `feat: R1Pro adapter, depth filter, nav/manip router`

**Gate:** tests green; filter rejects synthetic near-depth collision case.

---

### Task 5: Planner v0 (annotation classifier)

**Files:**
- Create: `src/wam/planner_v0.py`
- Create: `scripts/train_planner_v0.py`
- Create: `configs/planner_v0.yaml`
- Create: `reports/planner_v0_val.json`

**Interfaces:**
- `NextSkillPlanner.next_skill(obs, task) -> str`
- Train on skill windows; **no VLM** for gate

- [ ] Define feature: frozen visual embed stub (ResNet/Wan frozen encode) + language bag
- [ ] Train classifier on min subset; report top-1 next-skill accuracy on held-out windows
- [ ] Commit: `feat: planner v0 next-skill classifier`

**Gate:** `reports/planner_v0_val.json` with accuracy ≥ chance×3 on val windows (record threshold actually hit).

---

### Task 6: MoT-WAM skeleton (shapes + FAST forward)

**Files:**
- Create: `src/wam/encoders.py`, `src/wam/mot.py`, `src/wam/losses.py`, `src/wam/policy.py`
- Create: `configs/stage_a0.yaml`
- Test: `tests/test_mot_shapes.py`

**Interfaces:**
- `MoTWAM.forward_fast(batch) -> action_pred`
- `MoTWAM.forward_full(batch) -> (action_pred, world_pred)`
- `compute_losses(..., lambda_world) -> dict`

- [ ] Tests: batch shapes for FAST/FULL; `lambda_world=0` skips world grad path
- [ ] Implement minimal MoT (can start with thin Action Expert + hooked Wan blocks; full shared-attn MoT iteratively)
- [ ] Load Wan weights per `model_lock`; freeze early layers
- [ ] Commit: `feat: MoT-WAM FAST/FULL skeleton`

**Gate:** `pytest tests/test_mot_shapes.py` PASS on 1 GPU smoke batch.

---

### Task 7: Stage A0 train (action-only, FAST)

**Files:**
- Create: `scripts/train_stage.py`
- Modify: `configs/stage_a0.yaml`
- Create: `reports/checkpoint_selection_a0.json`

**Interfaces:**
- CLI: `python scripts/train_stage.py --stage a0`
- Checkpoint protocol: primary = val action loss (proxy until eval wired); save `checkpoints/best_a0.pt`

- [ ] Train A0 on skill episodes (manip+nav mixed)
- [ ] Log GPU-hours, steps, `backbone_forward_count`
- [ ] Commit: `feat: stage A0 training entrypoint`

**Gate:** `best_a0.pt` exists; val action loss trending down in `reports/`.

---

### Task 8: Hier-v0 vs Flat on public subset

**Files:**
- Create: `scripts/eval_subset.py`
- Create: `scripts/report_bootstrap.py`
- Modify: `configs/eval_public10.yaml` (fill 5 tasks × 10 instances)
- Create: `reports/hier_v0_bootstrap.json`

**Interfaces:**
- Flat: task language → MoT-WAM (ignore planner)
- Hier: Planner v0 → MoT-WAM + router + filter
- Legal completion: timeout + optional terminator stub

- [ ] Document power check (tasks×trials) in report header
- [ ] Run Flat and Hier; compute mean Q-score + bootstrap CI
- [ ] Commit: `report: hier_v0 vs flat gate`

**Gate:** Hier ΔQ ≥ 0.05 with CI lower > 0 **or** written failure analysis + fix iteration (do not proceed to A1c/B until Hier ≥ Flat or explicitly waived in `reports/hier_v0_bootstrap.json`).

---

### Task 9: A1c vs B (locked-same)

**Files:**
- Create: `configs/stage_a1c.yaml`, `configs/stage_b.yaml`
- Create: `reports/ablation_a1c_vs_b.md`, `reports/bootstrap_ci.json`, `reports/training_meta.json`

**Interfaces:**
- Same CLI `train_stage.py --stage a1c|b`
- Only difference: `lambda_world` 0 vs 0.3

- [ ] Train A1c and B with locked-same table; assert forward_count within 2%
- [ ] Eval on held-out / new-scene-heavy subset
- [ ] Bootstrap B−A1c; per-task drop check
- [ ] Commit: `report: a1c vs b world-loss ablation`

**Gate:** B vs A1c ΔQ ≥ 0.03 and CI lower > 0, per-task drop ≤ 0.05 — else follow rollback (data/mask/align) before scaling.

---

### Task 10: Stage C + submission pack

**Files:**
- Create: `configs/stage_c.yaml`
- Create: `scripts/pack_submission.sh`
- Create: `submission/README.md`

**Interfaces:**
- Pack: metrics JSONs, wrapper `.py`, robot yaml, README with exact eval CLI

- [ ] Stage C: FAST-heavy mix, `lambda_world=0.1` on FULL only; target+failure data
- [ ] Eval public validation; optionally expand task coverage
- [ ] Report π0.5/GR00T on same subset (external numbers table)
- [ ] `pack_submission.sh` dry-run
- [ ] Commit: `feat: stage C and submission pack`

**Gate:** dry-run pack complete; `git status` clean except data/ckpt ignores.

---

## Schedule (landable)

| Week | Tasks | Exit gate |
|------|-------|-----------|
| W0 | 0–2 | `model_lock` frozen + 1×1 smoke ok |
| W1 | 3–5 | verified skill data + planner v0 + filter tests |
| W2 | 6–8 | MoT A0 + Hier vs Flat gate |
| W3 | 9–10 | A1c vs B gate + Stage C pack |

**Resource downgrade order:** keep A1c+B → delay align ablations → delay Perceiver width / multi-seed.

## Explicitly out of scope (this plan)

- Separate SLAM / topological navigator
- Test-time future video denoising
- VLM planner for Hier gate
- Full 100-task training before public-subset gates pass

## Exec notes

1. Prefer one task → tests → commit cycles.
2. Large demos/checkpoints stay on high-capacity disk; only JSON reports in git.
3. After Task 8 fail: fix planner/completion/data before touching world loss.
4. After Task 9 fail: do not jump to 14B; fix locked-same / skill mix / depth stem first.

---

# 文档结束
