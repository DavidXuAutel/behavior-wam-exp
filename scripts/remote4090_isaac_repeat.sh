#!/usr/bin/env bash
# Measure Isaac SimulationApp startup flakiness on the 4090.
set -uo pipefail
N=${1:-8}
OUT=/home/a26125/behavior_logs/isaac_repeat
mkdir -p "$OUT"

# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset CUDA_VISIBLE_DEVICES

pass=0
for i in $(seq 1 "$N"); do
  log=$OUT/run_$i.log
  timeout 300 python - >"$log" 2>&1 <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
PY
  if grep -q APP_CREATED "$log"; then
    pass=$((pass+1)); echo "run $i: PASS"
  else
    sig=$(grep -m1 -oE "librtx[a-z.]*|libcarb[a-z.-]*|Crash detected in pid [0-9]+ thread [0-9]+" "$log" | head -1)
    echo "run $i: FAIL  ${sig:-no-crash-signature}"
  fi
done
echo "SUMMARY: $pass/$N passed"
