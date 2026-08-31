#!/bin/bash
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=1

LOG=/home/a25689/behavior-wam-exp/reports/behavior_env_setup.log
STATUS=/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json
ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K
OUT=/home/a25689/behavior-wam-exp/reports/recover_eula.out

{
  echo ""
  echo "===== EULA+DATASET RESUME $(date -u) ====="
} | tee -a "$LOG" "$OUT"

mkdir -p /home/a25689/.nvidia-omniverse/config
cat > /home/a25689/.nvidia-omniverse/config/privacy.toml <<'EOF'
[privacy]
performance = true
personalization = true
usage = true
EOF

echo "=== import check (pipe Yes) ===" | tee -a "$LOG" "$OUT"
set +e
printf 'Yes\n' | /opt/conda/bin/conda run -n behavior --no-capture-output \
  python -c 'import isaacsim, omnigibson, cv2; print("IMPORT_OK", cv2.__version__)' \
  2>&1 | tee -a "$LOG" "$OUT"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status": "blocked_eula",
  "message": "combined import failed after Yes pipe",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY
  exit 2
fi

python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status": "recovering",
  "message": "imports OK; resuming --dataset --eval",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY

echo "=== resume setup.sh --dataset --eval ===" | tee -a "$LOG" "$OUT"
cd "$ROOT"
# Run in background so this script can exit after launch confirmation
nohup /opt/conda/bin/conda run -n behavior --no-capture-output bash -lc '
  set -e
  cd /home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K
  export PATH=/opt/conda/envs/behavior/bin:/opt/conda/bin:$PATH
  export OMNIGIBSON_HEADLESS=1
  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y
  # Accept any residual EULA prompts
  printf "Yes\nYes\nYes\n" | ./setup.sh --dataset --eval \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
' >>"$LOG" 2>&1 &
RESUME_PID=$!
echo "RESUME_PID=$RESUME_PID" | tee -a "$LOG" "$OUT"

# Quick health: wait 20s and ensure process still alive
sleep 20
if kill -0 "$RESUME_PID" 2>/dev/null; then
  python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status": "running_dataset",
  "message": "dataset+eval setup resumed pid=$RESUME_PID",
  "resume_pid": $RESUME_PID,
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY
  echo "dataset resume healthy" | tee -a "$OUT"
else
  echo "resume exited early; see log" | tee -a "$OUT"
  tail -40 "$LOG" | tee -a "$OUT"
  exit 3
fi
