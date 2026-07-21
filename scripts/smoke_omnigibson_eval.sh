#!/usr/bin/env bash
# Task 2: OmniGibson eval smoke — fill OMNIGIBSON_ROOT after install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="${ROOT}/reports"
mkdir -p "${REPORT_DIR}"

: "${OMNIGIBSON_ROOT:?Set OMNIGIBSON_ROOT to your OmniGibson checkout}"

echo "[smoke] OmniGibson root: ${OMNIGIBSON_ROOT}"
echo "[smoke] TODO: invoke official eval for 1 task x 1 instance"
echo "[smoke] Write metrics summary to ${REPORT_DIR}/smoke_eval.json"

cat > "${REPORT_DIR}/smoke_eval.json" <<EOF
{
  "status": "not_run",
  "reason": "OmniGibson eval command not wired yet (Task 2)",
  "observation_legal": true,
  "independent_wam": true
}
EOF

echo "[smoke] placeholder written. Replace when eval is ready."
