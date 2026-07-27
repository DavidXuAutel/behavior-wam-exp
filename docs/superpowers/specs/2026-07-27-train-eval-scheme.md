# BEHAVIOR MoT-WAM — 完整训练与测评方案

> Date: 2026-07-27  
> Status: Active  
> Repo: `behavior-wam-exp`  
> 对齐：设计规格 rev2 · 可落地实施计划 · Challenge-first / Deployment-ready  
> 权威配套：`docs/pass_criteria.yaml` · `configs/model_lock.yaml`

**Goal:** 用官方 BEHAVIOR 数据完成可复现的主模型与公平消融；用 on-policy 失败/恢复数据提升鲁棒性；用 held-out scene + 扰动套件验证场景泛化，并为真机适配预留接口——不以真机数据阻塞 Challenge 主路径。

---

## 0. 总原则

| 原则 | 含义 |
|------|------|
| Challenge-first | A0 / Planner / Hier / A1c / B **只用官方 demo**，不混外部机器人数据 |
| 结构归因 | 未过官方数据门禁前，禁止用外部数据「刷分」掩盖架构问题 |
| 失败后补数 | A0 可闭环后再采 on-policy failure/recovery，仅进 Stage C（或明确标注的补强实验） |
| 评测隔离 | train / val / held-out **按 episode（整条轨迹）切分**，禁止技能段跨集合泄漏 |
| 合法观测 | 策略输入仅 RGB + Depth + proprio + 语言；禁止 BDDL / Q-score 在线输入 |
| 适配预留 | Sensor / Proprio / Action Adapter + embodiment token；当前只实现 R1Pro |

```text
Phase D0  数据审计与切分 ──► 不过则不训
Phase D1  官方数据主训（Planner → A0 → Hier → A1c/B）
Phase D2  闭环采失败/恢复 + 扰动 replay
Phase D3  Stage C 微调
Phase E   测评套件（Clean / Held-out Scene / Perturbed）+ 提交包
```

---

## 1. 数据方案

### 1.1 数据源角色

| 源 | 用途 | 阶段 |
|----|------|------|
| `2026-challenge-demos`（LeRobot V3） | Planner、A0、A1c、B、主消融 | D1 |
| 官方 raw HDF5 replay（可选） | 同轨迹重渲染扰动 | D2 |
| On-policy OmniGibson rollouts | 成功补充 + **失败/恢复** | D2→D3 |
| π0.5 / GR00T | **仅同子集外部对照**，不参与训练 | E |
| 真机数据 | **本方案不阻塞**；Challenge 后可选 | 未来 |

### 1.2 技能段样本格式

```python
{
  "episode_id": str,
  "task_id": str,
  "scene_id": str,
  "instance_id": str,
  "split": "train" | "val" | "heldout_scene",
  "rgb": ...,                 # multi-view, locked resolution
  "depth": ...,               # metric, required
  "proprio": ...,             # canonical schema
  "task_language": str,
  "skill": str,               # 31 skills
  "skill_class": "navigation" | "manipulation",
  "actions": ...,             # canonical action chunk
  "future_rgb": ...,          # FULL only
  "action_supervision": bool, # True only if verified demo / recovery policy-ok
  "video_supervision": bool,  # True for demos + most rollouts
  "terminator": 0 | 1,
  "source_id": "behavior_official" | "on_policy_success" | "failure_recovery" | "perturbed_replay",
  "embodiment_id": "r1pro",
}
```

### 1.3 切分协议（防泄漏）

```yaml
unit: episode          # 整条轨迹原子切分
primary_keys: [task_id, scene_id, episode_id]

# 建议比例（可按 scene 数微调，但规则固定）
train: 0.70            # 官方 episodes
val: 0.15              # 同 scene 未见 instance / 未见 episode
heldout_scene: 0.15    # 完整未见 scene（或官方标注的新场景任务集）

rules:
  - 同一 episode 的所有技能段必须落在同一 split
  - A1c vs B 的 held-out 评测优先用 heldout_scene
  - Stage C 采的 on-policy 数据不得覆盖 heldout_scene 的评测 instance
  - 提交用官方 evaluator 规定的 instance；内部门禁用本切分
```

产出文件：

- `data/splits/split_manifest.json`（每个 episode → split）
- `reports/data_audit.json`（§1.4）

### 1.4 开训前数据审计（D0 Gate）

脚本：`scripts/audit_behavior_data.py` → `reports/data_audit.json`

| 检查项 | 门槛 |
|--------|------|
| 主评测任务有效 demo 数 | ≥ 100 / task（不足则标记 oversample 或暂移出主子集） |
| 每技能技能段数 | ≥ 500；不足 → 过采样权重写入 `configs/sample_weights.yaml` |
| Navigation 占比 | 训练 batch 目标 **20–30%**（通过采样权重实现） |
| 相机 / Depth 缺失率 | < 1%；Depth 缺失段丢弃 |
| `action_dim` 与 `r1pro` 一致 | 100 条抽检全部通过 |
| 夹爪方向 / 限位 | 100 条抽检全部通过 |
| train/val/heldout 泄漏 | episode 交集为空 |
| 稀有技能列表 | 记录 top 稀有 5 个及权重 |

**Gate：** `reports/data_audit.json` 中 `pass: true`，否则禁止启动 A0。

### 1.5 训练混合比

#### Phase D1（官方 only）

```yaml
behavior_official_train: 1.0
# 其中采样：
nav_skill_target_fraction: 0.25
rare_skill_oversample: true
```

增强（在线，不改源文件）：

```yaml
aug:
  color_jitter: true
  depth_noise: true
  camera_extrinsic_jitter_small: true
  action_scale_jitter: [0.95, 1.05]
  control_delay_steps: [0, 1, 2]   # 训练时随机
```

#### Phase D3（Stage C）

```yaml
official_demos: 0.75
on_policy_success: 0.10
failure_and_recovery: 0.10   # L_action 仅 action_supervision=true 的恢复段
perturbed_rollouts: 0.05     # 主要 L_world；动作需通过质量过滤才 L_action
```

失败段监督规则：

| 段类型 | `L_world` | `L_action` |
|--------|-----------|------------|
| 官方成功 demo | ✓ | ✓ |
| on-policy 成功 | ✓ | ✓ |
| 失败过程（撞墙/空抓） | ✓ | ✗ |
| 人工/启发式标注的恢复动作 | ✓ | ✓ |
| 纯随机扰动且动作不可信 | ✓ | ✗ |

---

## 2. 训练管线（按阶段）

### 2.0 公共前置

1. `configs/model_lock.yaml` → `status: frozen`
2. OmniGibson 1×1 smoke OK
3. D0 数据审计 `pass: true`
4. Canonical adapters 单测通过（obs / proprio / action）

耦合主路径（A0 钉死，与结构 review 一致）：

```text
RGB → Wan VAE → Video Expert（A0 冻；A1c/B LoRA）
Depth → shallow stem
Proprio / skill / text → tokens
         → Perceiver(64) → Action FM-DiT
FULL: + future video latent FM loss
FAST: 无未来视频 token（部署同款）
```

### 2.1 Planner v0

| 项 | 设定 |
|----|------|
| 输入 | 冻结视觉 embed + 任务语言 + 上一技能 |
| 输出 | 下一技能（31 类）+ 可选 terminator |
| 数据 | 仅官方 train 技能段 |
| Loss | CE(skill) + optional BCE(terminator) |
| 禁止 | 在线 VLM；BDDL/Q-score 特征 |
| 产物 | `checkpoints/planner_v0.pt` · `reports/planner_v0_val.json` |

**Gate：** val top-1 ≥ `3 × chance`（chance≈1/31）；记录实际准确率。

### 2.2 Stage A0（action-only FAST）

| 项 | 设定 |
|----|------|
| 可训 | Depth stem, proprio/skill encoders, Perceiver, Action Expert |
| 冻结 | Wan VAE, text encoder, Video Expert 全部权重 |
| 模式 | FAST only |
| Loss | `L_action` only |
| 数据 | Phase D1 官方 train |
| 选点 | val action loss 最低（辅：技能段成功率代理若有） |
| 产物 | `checkpoints/best_a0.pt` |

**Gate：**

- val `L_action` 相对初始下降 ≥ 30%（或平台记录趋势）
- 烟雾闭环：公共子集上 Q-score **高于 zero-action / random**（定性）
- 导航技能段：base 维预测 MSE 不差于手臂维的相对水平（防「完全不会走」）

### 2.3 Hier-v0 vs Flat（训练后第一评测门）

| 臂 | 定义 |
|----|------|
| Flat | 任务语言 → A0 WAM；无 Planner |
| Hier | Planner v0 → A0 WAM → router → filter |

数据：`configs/eval_public10.yaml`（建议 5 tasks × 10 instances × 1 rollout）  
指标：mean Q-score + bootstrap 95% CI  
**Gate：** Hier − Flat ≥ **0.05** 且 CI 下界 > 0；否则修 Planner/完成检测/数据，**不开 A1c/B**（除非 `reports/hier_v0_bootstrap.json` 书面豁免）。

### 2.4 Stage A1c vs B（公平消融）

Locked-same（唯一差异 `λ_w`）：

| 配置 | A1c | B |
|------|-----|---|
| 数据 / 步数 / LR / LoRA / FULL:FAST | 相同 | 相同 |
| FULL:FAST | 0.5 : 0.5 | 0.5 : 0.5 |
| `λ_w` | **0** | **0.3** |
| Video Expert | LoRA | LoRA |
| Action / Perceiver | 全训 | 全训 |

Loss：

- FAST: `L = L_action`
- FULL A1c: `L = L_action`
- FULL B: `L = L_action + 0.3 L_world`

日志强制：`backbone_forward_count`、GPU-hours；二者 forward_count 差 ≤ 2%。

选点：以 **heldout_scene Q-score**（或 perturbed 代理）为主，val `L_action` 为辅。

**Gate：**

- B − A1c ≥ **0.03** Q-score on held-out scenes，CI 下界 > 0
- 任一任务相对 A1c 掉点 ≤ **0.05**
- 不过：按回退树修数据权重 / depth stem / train-test 对齐；**禁止直接上 14B**

### 2.5 Phase D2：闭环采数（A0 或 B 可跑后）

采集脚本：`scripts/collect_on_policy.py`

覆盖失败模式（每类至少目标条数，写入报告）：

- 导航走偏 / 碰撞 / 超时
- 空抓、掉落
- 门/抽屉未打开
- place / pour 未达成
- 从失败态恢复（若可启发式或短遥操标注）

同时：`scripts/replay_perturb.py` 对官方轨迹做轻量重渲染扰动（光照、小相机抖、depth 噪声）。

### 2.6 Stage C

| 项 | 设定 |
|----|------|
| 初始化 | `best_B`（若 B 未过门则 `best_A1c` 或 `best_A0`，须在报告写明） |
| 混合 | §1.5 Phase D3 |
| 模式 | FULL:FAST = **0.2 : 0.8** |
| Loss | FAST: `L_action`；FULL: `L_action + 0.1 L_world` |
| 冻结建议 | VAE / text 冻；Video LoRA 可小 LR；Action+Perceiver 主训 |

**Gate：** Clean 公共子集 Q-score ≥ Stage B（或 A1c）同子集；Perturbed 相对 Clean 掉点 ≤ **15%**（§3.3）。

---

## 3. 测评方案

### 3.1 测评套件一览

| 套件 ID | 目的 | 环境 | 规模（内部门禁建议） |
|---------|------|------|----------------------|
| **E0 Smoke** | 管道合法 | 1 task × 1 inst | 必过 |
| **E1 Clean Public** | 主进度 / Hier / C | 5–10 tasks × 10 inst | 门禁 |
| **E2 Held-out Scene** | 场景泛化 / A1c vs B | heldout scenes × 固定 inst | 门禁 |
| **E3 Perturbed** | 部署向鲁棒 | E1 任务 + 扰动因子 | 门禁 |
| **E4 External Baseline** | 对照 π0.5/GR00T | 与 E1 **同子集同协议** | 报告 |
| **E5 Challenge Full** | 提交 | 100×10×1 官方规定 | 最终 |

所有套件：

- 每 instance **1 rollout**（与提交一致；禁止 cherry-pick best-of-N）
- 固定随机种子表写入 `reports/eval_seeds.json`
- 策略侧零特权信息

### 3.2 扰动因子（E3）

每次 episode 采样一个组合（写入 metrics JSON）：

```yaml
perturbations:
  lighting: {gain: [0.7, 1.3], color_temp_shift: small}
  camera_extrinsic: {trans_m: 0.01, rot_deg: 2}
  depth_noise: {gaussian_std_m: 0.01, dropout: 0.02}
  control_delay_steps: [0, 1, 2]
  action_scale: [0.9, 1.1]
  # 可选（算力允许）:
  friction_scale: [0.8, 1.2]
```

报告：Clean vs Perturbed 的 ΔQ 与 95% CI。

### 3.3 指标与通过线

主指标：**mean Q-score**（BDDL 部分给分）  
辅指标：满段成功率、平均完成时间、技能切换次数、碰撞/超时计数、推理 p50/p95 延迟

写入 `docs/pass_criteria.yaml` 的扩展目标：

| 对比 | 门槛 |
|------|------|
| Hier − Flat（E1） | ≥ +0.05，CI 下界 > 0 |
| B − A1c（E2） | ≥ +0.03，CI 下界 > 0 |
| Per-task vs A1c | 掉点 ≤ 0.05 |
| E2 vs E1（同模型） | 相对掉点 ≤ **10%**（场景泛化） |
| E3 vs E1（同模型） | 相对掉点 ≤ **15%**（扰动鲁棒） |
| 推理频率 | ≥ `model_lock.control.target_hz`（通常 3–5 Hz） |

Power check：在声称门禁通过前，于 `reports/power_check.json` 记录子集规模与可检测效应量。

### 3.4 评测流程（单次）

```text
加载 robot profile + model_lock
→ pack_obs（强制 depth）
→ HierarchicalPolicy 或 Flat
→ FAST infer → router → filter → R1Pro adapter
→ OmniGibson evaluator
→ 写 metrics JSON（含 suite_id / perturb_id / seed）
→ 聚合 scripts/report_bootstrap.py
```

技能完成（策略侧）：超时 / terminator / 合法启发式；**绝不**读 evaluator Q-score。

### 3.5 报告产物（必须落盘）

```text
reports/
  data_audit.json
  split_manifest_summary.json
  planner_v0_val.json
  checkpoint_selection_{a0,a1c,b,c}.json
  hier_v0_bootstrap.json
  ablation_a1c_vs_b.md
  bootstrap_ci.json
  training_meta.json          # forward_count, gpu_hours, λ_w
  eval_e1_clean.json
  eval_e2_heldout_scene.json
  eval_e3_perturbed.json
  eval_e4_baselines.json
  power_check.json
  generalization_summary.md   # E2/E1、E3/E1 掉点表
```

---

## 4. 阶段时间表（与落地计划对齐）

| 周 | 训练 | 测评 | 硬门禁 |
|----|------|------|--------|
| W0 | model_lock + smoke | E0 | lock frozen + E0 ok |
| W1 | D0 审计 + 切片 + Planner + filter | Planner val | audit pass |
| W2 | A0 → Hier vs Flat | E1 | Hier gate |
| W2–3 | A1c / B | E1 + **E2** | B vs A1c gate |
| W3 | D2 采数 + Stage C | E1 + E2 + **E3** + E4 | 泛化掉点门槛 + pack |

资源不足时：**保 E1+E2 上的 A1c/B** → 推迟 E3 全因子 → 推迟 E4 完整对照。

---

## 5. 配置与代码落点

```text
configs/
  data_splits.yaml           # 切分规则
  sample_weights.yaml        # nav / 稀有技能
  stage_a0.yaml | a1c.yaml | b.yaml | c.yaml
  eval_e1_clean.yaml
  eval_e2_heldout_scene.yaml
  eval_e3_perturbed.yaml
  eval_e4_baselines.yaml
scripts/
  audit_behavior_data.py
  build_skill_episodes.py
  train_planner_v0.py
  train_stage.py
  collect_on_policy.py
  replay_perturb.py
  eval_subset.py
  report_bootstrap.py
  pack_submission.sh
```

---

## 6. 明确不做（本方案周期内）

- 用非 BEHAVIOR 本体数据训主模型
- 真机采集阻塞 Challenge 时间线
- 测试时未来视频去噪
- VLM Planner 作为 Hier 门禁依赖
- 未过 D0/A0/Hier 前扩到 100 任务全量重训

---

## 7. 一句话执行口令

**官方数据把结构做对并过 Hier 与 A1c/B；闭环失败数据只服务 Stage C；Clean / Held-out Scene / Perturbed 三套测评决定能否扩规模与提交。**
