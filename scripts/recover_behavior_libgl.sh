#!/bin/bash
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
LOG=/home/a25689/behavior-wam-exp/reports/behavior_env_setup.log
STATUS=/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json
ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K

{
  echo ""
  echo "===== MANUAL RECOVERY $(date -u) ====="
} >>"$LOG"

echo "=== kill stuck interactive setup prompts ==="
pkill -f 'setup.sh --omnigibson' 2>/dev/null || true
pkill -f 'setup.sh --new-env behavior' 2>/dev/null || true
sleep 1

echo "=== probe packages ==="
/opt/conda/bin/conda run -n behavior python -c 'import importlib.util as u; print("isaacsim", bool(u.find_spec("isaacsim"))); print("omnigibson", bool(u.find_spec("omnigibson")))' 2>&1 | tee -a "$LOG"

echo "=== install libGL via conda-forge into behavior ==="
# Provide libGL.so.1 without system sudo
/opt/conda/bin/conda install -y -n behavior -c conda-forge libgl mesa-libgl-cos7-x86_64 2>&1 | tee -a "$LOG" || \
/opt/conda/bin/conda install -y -n behavior -c conda-forge libgl 2>&1 | tee -a "$LOG" || true

# Also try system apt if passwordless sudo works
if sudo -n true 2>/dev/null; then
  sudo -n apt-get update -y 2>&1 | tee -a "$LOG" || true
  sudo -n apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 2>&1 | tee -a "$LOG" || true
fi

echo "=== find libGL ==="
find /opt/conda/envs/behavior /usr/lib /lib -name 'libGL.so*' 2>/dev/null | head -30 | tee -a "$LOG"

# Prefer conda lib path
CONDA_LIB=/opt/conda/envs/behavior/lib
export LD_LIBRARY_PATH="${CONDA_LIB}:${CONDA_LIB}/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# NVIDIA GLX often enough with libGLdispatch
if [ -e /lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] && [ ! -e /lib/x86_64-linux-gnu/libGL.so.1 ]; then
  echo "NVIDIA GLX present but libGL.so.1 missing" | tee -a "$LOG"
fi

echo "=== retry cv2 / omnigibson import ==="
OMNIGIBSON_HEADLESS=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /opt/conda/bin/conda run -n behavior --no-capture-output \
  python -c 'import cv2; print("cv2", cv2.__version__); import omnigibson; print("og_ok")' 2>&1 | tee -a "$LOG" || true

# If still failing, force opencv headless reinstall
if ! OMNIGIBSON_HEADLESS=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /opt/conda/bin/conda run -n behavior python -c 'import cv2' >/dev/null 2>&1; then
  echo "=== reinstall opencv-python-headless ===" | tee -a "$LOG"
  /opt/conda/bin/conda run -n behavior python -m pip uninstall -y opencv-python opencv-contrib-python 2>&1 | tee -a "$LOG" || true
  /opt/conda/bin/conda run -n behavior python -m pip install -U opencv-python-headless 2>&1 | tee -a "$LOG"
fi

echo "=== final import check ==="
set +e
OMNIGIBSON_HEADLESS=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /opt/conda/bin/conda run -n behavior --no-capture-output \
  python -c 'import isaacsim, omnigibson, cv2; print("IMPORT_OK", cv2.__version__)' 2>&1 | tee -a "$LOG"
IMPORT_RC=${PIPESTATUS[0]}
set -e

if [ "$IMPORT_RC" -eq 0 ]; then
  echo "=== resume dataset+eval install inside behavior env ===" | tee -a "$LOG"
  cd "$ROOT"
  # Activate properly via conda run bash
  /opt/conda/bin/conda run -n behavior --no-capture-output bash -lc '
    set -e
    cd /home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K
    export OMNIGIBSON_HEADLESS=1
    export PATH=/opt/conda/envs/behavior/bin:/opt/conda/bin:$PATH
    # Only remaining pieces if isaac/og already present
    ./setup.sh --dataset --eval --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
  ' >>"$LOG" 2>&1 &
  echo "RESUME_PID=$!" | tee -a "$LOG"
  python3 - <<PY
import json, datetime
open("/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json","w").write(json.dumps({
  "status":"recovering",
  "message":"libGL/cv2 fixed; resumed --dataset --eval",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY
else
  python3 - <<PY
import json, datetime
open("/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json","w").write(json.dumps({
  "status":"blocked_libgl",
  "message":"import still fails after libGL/opencv recovery attempts",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY
  exit 2
fi
