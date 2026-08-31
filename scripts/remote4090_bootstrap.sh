#!/usr/bin/env bash
# Bootstrap Isaac/OmniGibson on 10.229.20.110 (RTX 4090)
set -euo pipefail

LOGDIR=/home/a26125/behavior_logs
MNT=/home/a26125/behavior_mnt
KEY=/home/a26125/.ssh/id_ed25519_behavior
H100=a25689@10.239.121.22
H100_PORT=31103
H100_BEHAVIOR=/home/a25689/BEHAVIOR-2026
STATUS=$LOGDIR/bootstrap_status.json

mkdir -p "$LOGDIR" "$MNT" /home/a26125/behavior-wam-exp
exec > >(tee -a "$LOGDIR/bootstrap.log") 2>&1
echo "======== $(date -Is) start ========"
df -h /home | tail -1
nvidia-smi -L

write_status() {
  printf '{"status":"%s","ts":"%s","detail":"%s"}\n' "$1" "$(date -Is)" "$2" >"$STATUS"
}

# Free regenerable Carla leftover content (~9.6G) if still present
if [[ -d /home/a26125/_carla_content_ue5 ]]; then
  echo "Removing /home/a26125/_carla_content_ue5 to free space for Isaac..."
  rm -rf /home/a26125/_carla_content_ue5
fi
df -h /home | tail -1

# conda sshfs
# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate base
if ! command -v sshfs >/dev/null 2>&1; then
  write_status installing_sshfs "conda-forge sshfs"
  conda install -y -c conda-forge sshfs
fi
command -v sshfs
command -v fusermount3 || command -v fusermount

# Mount H100 BEHAVIOR tree
if ! mountpoint -q "$MNT"; then
  write_status mounting "sshfs ${H100}:${H100_BEHAVIOR}"
  sshfs -o IdentityFile="$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,allow_other=false \
    -p "$H100_PORT" "${H100}:${H100_BEHAVIOR}" "$MNT"
fi
mountpoint -q "$MNT"
echo "MOUNT_OK"
ls "$MNT"
ls "$MNT/code" 2>/dev/null || true
ls "$MNT/datasets" 2>/dev/null || true

CODE=$MNT/code/BEHAVIOR-1K
if [[ ! -d "$CODE" ]]; then
  write_status error "BEHAVIOR-1K missing on mount"
  exit 30
fi

# Dataset paths on Ceph (via mount). Prefer BEHAVIOR-1K/datasets if that is where assets live.
if [[ -d "$CODE/datasets" ]]; then
  export OMNIGIBSON_DATA_PATH="$CODE/datasets"
elif [[ -d "$MNT/datasets/behavior" ]]; then
  export OMNIGIBSON_DATA_PATH="$MNT/datasets/behavior"
else
  export OMNIGIBSON_DATA_PATH="$MNT/datasets"
fi
echo "OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH"
du -sh "$OMNIGIBSON_DATA_PATH" 2>/dev/null | head || true
ls "$OMNIGIBSON_DATA_PATH" 2>/dev/null | head || true

# Create behavior env (no --dataset; assets via mount)
if conda env list | awk '{print $1}' | grep -qx behavior; then
  echo "behavior env exists"
  write_status env_exists "skip create"
else
  write_status installing_env "setup.sh --new-env behavior (no dataset)"
  cd "$CODE"
  # Match H100 .22 flags from docs/REMOTE_H100_22.md
  printf 'Yes\nYes\nYes\n' | ./setup.sh --new-env behavior --omnigibson --bddl --joylo --eval \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos \
    2>&1 | tee "$LOGDIR/setup_behavior.log"
fi

write_status env_ready "conda env behavior"
df -h /home | tail -1
conda env list | grep behavior || true

# Quick Isaac gate (must print APP_CREATED)
write_status isaac_smoke "SimulationApp headless"
# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
timeout 180 python - <<'PY' | tee "$LOGDIR/isaac_app_created.txt"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
PY

if grep -q APP_CREATED "$LOGDIR/isaac_app_created.txt"; then
  write_status success "APP_CREATED"
else
  write_status failed_isaac "no APP_CREATED"
  exit 40
fi

echo "======== $(date -Is) done ========"
