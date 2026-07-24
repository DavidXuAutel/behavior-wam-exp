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
  model_lock.yaml              # Week 0 freeze
  skill_set.yaml
  data_compatibility.yaml
  eval_public10.yaml
  stage_a0.yaml | stage_a1c.yaml | stage_b.yaml | stage_c.yaml
src/wam/
  obs_pack.py                  # legal obs packing + depth checks
  encoders.py                  # depth stem, state MLP, skill embed
  mot.py                       # MoT Video+Action experts
  losses.py                    # L_action, L_world, masks
  policy.py                    # FAST infer API
  planner_v0.py                # next-skill classifier
src/data/
  lerobot_behavior.py          # demo reader
  skill_slice.py               # nav/manip windows
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
reports/                       # JSON gates only
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
