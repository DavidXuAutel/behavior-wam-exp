#!/usr/bin/env bash
# Capture the real fault (signal + address) of the Isaac RTX crash under gdb.
set -uo pipefail
OUT=/home/a26125/behavior_logs/isaac_gdb.log

# shellcheck disable=SC1091
source /home/a26125/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset CUDA_VISIBLE_DEVICES

cat > /tmp/isaac_min.py <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("APP_CREATED", flush=True)
app.close()
PY

# Let Kit's own handler stay out of the way so gdb sees the first fault.
timeout 300 gdb -batch -nx \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex 'handle SIGSEGV stop nopass' \
  -ex run \
  -ex 'printf "\n===== FAULT =====\n"' \
  -ex 'info program' \
  -ex 'print $_siginfo._sifields._sigfault.si_addr' \
  -ex 'bt 25' \
  -ex 'info registers rip rsp rbp' \
  --args python /tmp/isaac_min.py >"$OUT" 2>&1

echo "exit=$?"
grep -nE "APP_CREATED|Program received|===== FAULT|si_addr|^#[0-9]" "$OUT" | head -40
