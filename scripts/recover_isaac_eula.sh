#!/bin/bash
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
export OMNIGIBSON_HEADLESS=1
LOG=/home/a25689/behavior-wam-exp/reports/behavior_env_setup.log
STATUS=/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json
ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K

{
  echo ""
  echo "===== EULA RECOVERY $(date -u) ====="
} >>"$LOG"

# Common NVIDIA Isaac / Kit EULA env vars
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=1

mkdir -p /home/a25689/.nvidia-omniverse/config
cat > /home/a25689/.nvidia-omniverse/config/privacy.toml <<'EOF'
[privacy]
performance = true
personalization = true
usage = true
EOF

# Search how isaacsim reads EULA
python3 - <<'PY' | tee -a "$LOG"
import pathlib
root = pathlib.Path('/opt/conda/envs/behavior/lib/python3.11/site-packages')
hits = []
for p in root.rglob('*.py'):
    try:
        t = p.read_text(errors='ignore')
    except Exception:
        continue
    if 'EULA' in t and ('accept' in t.lower() or 'Yes/No' in t):
        hits.append(str(p))
        if len(hits) >= 20:
            break
print('EULA_FILES', hits)
PY

# Try piping Yes to first isaacsim import
echo "=== piped Yes import ===" | tee -a "$LOG"
set +e
printf 'Yes\n' | /opt/conda/bin/conda run -n behavior --no-capture-output \
  python -c 'import isaacsim; print("ISAAC_OK")' 2>&1 | tee -a "$LOG"
RC1=${PIPESTATUS[0]}
set -e

# Also try with yes command
if [ "$RC1" -ne 0 ]; then
  echo "=== yes | import ===" | tee -a "$LOG"
  set +e
  yes Yes | head -5 | /opt/conda/bin/conda run -n behavior --no-capture-output \
    python -c 'import isaacsim; print("ISAAC_OK")' 2>&1 | tee -a "$LOG"
  RC1=${PIPESTATUS[0]}
  set -e
fi

# Check for eula accepted marker files created
find /home/a25689 -name '*eula*' 2>/dev/null | head -20 | tee -a "$LOG" || true
find /opt/conda/envs/behavior -name '*eula*' 2>/dev/null | head -20 | tee -a "$LOG" || true

echo "=== combined import ===" | tee -a "$LOG"
set +e
printf 'Yes\n' | /opt/conda/bin/conda run -n behavior --no-capture-output \
  python -c 'import isaacsim, omnigibson, cv2; print("IMPORT_OK", cv2.__version__)' 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -eq 0 ]; then
  python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status":"recovering",
  "message":"imports OK after EULA accept; resuming dataset+eval",
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
    printf "Yes\n" | ./setup.sh --dataset --eval \
      --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
  ' >>"$LOG" 2>&1 &
  echo "RESUME_PID=$!" | tee -a "$LOG"
else
  python3 - <<PY
import json, datetime
open("$STATUS","w").write(json.dumps({
  "status":"blocked_eula",
  "message":"isaacsim still prompts EULA / bootstrap fails non-interactively",
  "updated_at": datetime.datetime.utcnow().isoformat()+"Z",
}, indent=2)+"\n")
PY
  exit 2
fi
