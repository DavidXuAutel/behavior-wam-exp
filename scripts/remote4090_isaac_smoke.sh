#!/usr/bin/env bash
# Isaac SimulationApp gate on 4090
set -euo pipefail
LOG=/home/a26125/behavior_logs/isaac_app_created.txt
# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
# Do NOT set CUDA_VISIBLE_DEVICES — Isaac crashreporter warns it can crash.
unset CUDA_VISIBLE_DEVICES || true
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PATH=/home/a26125/miniconda3/envs/behavior/bin:$PATH
echo "python=$(command -v python)"
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
vulkaninfo --summary 2>&1 | grep -E 'deviceName|deviceType|apiVersion|driverVersion' | head -10
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "name", torch.cuda.get_device_name(0))'
timeout 300 python - <<'PY' | tee "$LOG"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
print("APP_CLOSED", flush=True)
PY
grep -q APP_CREATED "$LOG"
echo SMOKE_OK
