#!/bin/bash
# Watch BEHAVIOR setup; restart on unexpected death without success.
set -u
LOG=/home/a25689/behavior-wam-exp/reports/behavior_env_setup.log
STATUS=/home/a25689/behavior-wam-exp/reports/behavior_env_setup_status.json
ROOT=/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K
PIDFILE=/home/a25689/behavior-wam-exp/reports/watch_behavior_setup.pid
export PATH=/opt/conda/bin:$PATH
mkdir -p "$(dirname "$STATUS")"
echo $$ > "$PIDFILE"

write_status() {
  local st="$1"
  local msg="$2"
  python3 - "$st" "$msg" "$STATUS" <<'PY'
import json, datetime, sys
st, msg, path = sys.argv[1], sys.argv[2], sys.argv[3]
data = {
  "status": st,
  "message": msg,
  "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
}
open(path, "w").write(json.dumps(data, indent=2) + "\n")
print(json.dumps(data), flush=True)
PY
}

is_setup_running() {
  pgrep -f '/bin/bash ./setup.sh' >/dev/null \
    || pgrep -f 'setup.sh --new-env behavior' >/dev/null \
    || pgrep -f 'setup.sh --omnigibson' >/dev/null
}

imports_ok() {
  OMNIGIBSON_HEADLESS=1 /opt/conda/bin/conda run -n behavior \
    python -c 'import isaacsim, omnigibson; print("IMPORT_OK")' >/dev/null 2>&1
}

attempt=0
max_restart=2
idle_stall_sec=1800
last_size=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
last_change=$(date +%s)

while true; do
  now=$(date +%s)
  size=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "$size" != "$last_size" ]; then
    last_size=$size
    last_change=$now
  fi

  if is_setup_running; then
    if [ $((now - last_change)) -gt $idle_stall_sec ]; then
      write_status "stalled" "log unchanged ${idle_stall_sec}s; PID still alive"
    else
      write_status "running" "setup active; log_bytes=$size"
    fi
    sleep 120
    continue
  fi

  if imports_ok; then
    write_status "success" "isaacsim+omnigibson import OK"
    exit 0
  fi

  attempt=$((attempt + 1))
  if [ "$attempt" -le "$max_restart" ]; then
    write_status "restarting" "setup died; restart $attempt/$max_restart"
    {
      echo ""
      echo "===== WATCHDOG RESTART $attempt at $(date -u) ====="
    } >>"$LOG"
    cd "$ROOT" || exit 1
    if /opt/conda/bin/conda env list | grep -qE '^behavior[[:space:]]'; then
      nohup ./setup.sh --omnigibson --bddl --joylo --dataset --eval \
        --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos \
        >>"$LOG" 2>&1 &
    else
      nohup ./setup.sh --new-env behavior --omnigibson --bddl --joylo --dataset --eval \
        --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos \
        >>"$LOG" 2>&1 &
    fi
    sleep 45
    continue
  fi

  write_status "failed" "setup dead after $max_restart restarts; imports still fail"
  exit 1
done
