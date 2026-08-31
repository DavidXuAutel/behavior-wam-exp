#!/bin/bash
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

LOG=/home/a25689/behavior-wam-exp/reports/behavior_env_setup.log
STATUS=/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json
ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K

{
  echo ""
  echo "===== DATASET RESUME2 $(date -u) ====="
} >>"$LOG"

python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status": "running_dataset",
  "message": "resuming setup with --omnigibson --dataset --eval",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY

cd "$ROOT"
nohup /opt/conda/bin/conda run -n behavior --no-capture-output bash -lc '
  set -e
  cd /home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K
  export PATH=/opt/conda/envs/behavior/bin:/opt/conda/bin:$PATH
  export OMNIGIBSON_HEADLESS=1
  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y
  printf "Yes\nYes\nYes\n" | ./setup.sh --omnigibson --dataset --eval \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
' >>"$LOG" 2>&1 &
echo "RESUME_PID=$!"
sleep 15
if kill -0 $! 2>/dev/null || pgrep -f 'setup.sh --omnigibson --dataset' >/dev/null; then
  echo "running"
else
  echo "early_exit"
  tail -30 "$LOG"
fi
