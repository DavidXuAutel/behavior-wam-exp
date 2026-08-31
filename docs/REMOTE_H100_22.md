# Remote machine: 10.239.121.22 (2×H100)

**Previous node:** `10.239.121.21:31126` (`jpt-a25689-260727-fba7a-default0-0`) — same Ceph `/home/a25689`, but FastWAM depth jobs occupied both GPUs (~66 GiB/GPU), blocking E0. Switch to this node to unblock.

## Status (2026-07-28 — Vulkan ICD fixed; E0 blocked on Isaac RTX hang)

| Item | Value |
|------|--------|
| SSH | `ssh a25689@10.239.121.22 -p 31103` (expect: `PreferredAuthentications=password` + `PubkeyAuthentication=no`) |
| Connectivity | **ssh_ok** (host `jpt-a25689-260728-e8024-default0-0`) |
| GPUs | 2× NVIDIA H100 80GB HBM3 — **idle** during E0 attempt |
| Home storage | Ceph ~157T, ~89T free (same CSI volume as .21) |
| Official demos | `/home/a25689/BEHAVIOR-2026/datasets/2026-challenge-demos` (~3.0T) |
| Code workspace | `/home/a25689/behavior-wam-exp` |
| BEHAVIOR-1K / OmniGibson | `/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K` (+ `OmniGibson`) |
| Datasets (Ceph) | `behavior-1k-assets` **33G**; `omnigibson-robot-assets` **2.8G**; `2026-challenge-task-instances` **559M**; `omnigibson.key` present under `BEHAVIOR-1K/datasets/` |
| `mot-wam` / `kairos` | **untouched** |
| BEHAVIOR env `behavior` | **OK on this node** — `/opt/conda/envs/behavior` (~20G). Setup log: `reports/behavior_env_setup_22.log` (PID 898, exit 0). Flags: `--new-env behavior --omnigibson --bddl --joylo --eval` + TOS flags; **no `--dataset`**; do **not** combine `--confirm-no-conda` with `--new-env` (mutually exclusive). |
| Imports | **OK** after `sudo apt-get install -y libgl1 libglu1-mesa` (+ joylo editable reinstall). torch 2.7.0+cu128, 2 GPUs, omnigibson 3.9.0 |
| OmniGibson smoke (E0) | **failed_rtx_hang** — Vulkan ICD **fixed** (libEGL.so.1 via libegl1/libglvnd0); Kit starts with Graphics API Vulkan on both H100s. Then hangs forever in SimulationApp._prepare_ui / first app.update (rtx.materialdb). Minimal Isaac Sim headless reproduces. Logs: reports/e0_smoke_vkfix_22.log. |

## Shared vs node-local

| Layer | On .22? | Notes |
|-------|---------|--------|
| Ceph home (code, demos, assets, Wan symlink, reports, kairos, mot-wam) | **yes** | Same volume as .21 — no bulk copy needed |
| `/opt/conda/envs/behavior` | **yes** | Recreated 2026-07-28 (~9 min setup) |
| GPU freedom for E0 | **yes** | Idle; Vulkan OK; **Isaac RTX hangs after app ready** |

## Tier implication

2×H100 = **MVP / Base-lite**. Same as .21: RGB **256**, head (+ optional 1 wrist); `coupling: perceiver_cross_attn` for A0.

## Next commands (on remote .22)

```bash
# Vulkan sanity (should list 2x H100):
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary

# Gate for E0: must print APP_CREATED (currently HANGS on this node)
export PATH=/opt/conda/envs/behavior/bin:/opt/conda/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED")
app.close()
PY

# Only after APP_CREATED works:
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
bash ~/behavior-wam-exp/reports/run_e0_smoke_retry.sh
```

**Admin ask:** bake `libegl1 libglvnd0 libopengl0 libgles2 libglx0` into the image; fix Isaac Sim 5.1 / Kit 107.3 RTX hang on this H100 container (or provide an Isaac-validated node).

## Do not

- Overwrite `kairos`, `mot-wam`, or working training envs
- Mix FastWAM tree into this training path (Wan weights symlink only)
- Start A0 before `model_lock.status: frozen` and `reports/data_audit.json` pass
- Re-download ~36G assets unless integrity check fails (already on Ceph)
- Modify Franka / robot networks
