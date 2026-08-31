#!/usr/bin/env bash
# Isolate the Isaac RTX crash on 10.229.20.110 (RTX 4090).
# Runs a few variants and reports which one reaches APP_CREATED.
set -uo pipefail

E=/home/a26125/miniconda3/envs/behavior
KIT=$E/lib/python3.11/site-packages/isaacsim/kit
OUT=/home/a26125/behavior_logs
mkdir -p "$OUT"

# Caches were copied from the H100 node; drop anything GPU-specific.
rm -rf "$HOME/.cache/ov" "$HOME/.nvidia-omniverse/pycache" \
       "$KIT/cache/shadercache" "$KIT/cache/nv_shadercache" "$KIT/cache/DerivedDataCache" \
       "$KIT/data/Kit/Isaac-Sim Python/5.1/"*.dmp 2>/dev/null

# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset CUDA_VISIBLE_DEVICES

run_variant() {
  local name=$1 log=$OUT/isaac_try_$1.log
  shift
  echo "######## variant=$name ########"
  ( "$@" ) >"$log" 2>&1
  if grep -q APP_CREATED "$log"; then
    echo "RESULT $name = PASS"
  else
    echo "RESULT $name = FAIL ($(grep -cE '\[Fatal\]' "$log") fatal lines)"
    grep -m3 -E "Crash detected|段错误|core dumped|Fatal.*!" "$log" | cut -c1-160
  fi
}

v_default() {
  timeout 300 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
PY
}

v_smallviewport() {
  timeout 300 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 128, "height": 128, "hide_ui": True})
print("APP_CREATED", flush=True)
app.close()
PY
}

v_nolayers() {
  VK_LOADER_LAYERS_DISABLE='*' timeout 300 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
PY
}

v_noviewport() {
  # Skip the viewport/scenedb path that crashes; enough to prove Kit+RTX loads.
  timeout 300 python - <<'PY'
import sys
sys.argv += ["--no-window", "--/app/asyncRendering=false",
             "--/rtx/rendermode=RaytracedLighting",
             "--/app/renderer/skipWhileMinimized=false"]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "create_new_stage": False})
print("APP_CREATED", flush=True)
app.close()
PY
}

run_variant default       v_default
run_variant smallviewport v_smallviewport
run_variant nolayers      v_nolayers
run_variant noviewport    v_noviewport
echo "######## done ########"
