#!/usr/bin/env bash
# Run on 10.229.20.110 as a26125
set -euo pipefail

mkdir -p /home/a26125/behavior-wam-exp /home/a26125/behavior_mnt /home/a26125/behavior_logs
REPORT=/home/a26125/behavior_logs/probe_mount.txt
{
  echo "=== $(date -Is) probe ==="
  nvidia-smi -L
  df -h /home / | head -5
  echo "--- H100 ssh ---"
} >"$REPORT"

# Non-interactive H100 check via sshpass if available, else expect-less password file
export SSHPASS=123456
if command -v sshpass >/dev/null 2>&1; then
  sshpass -e ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no -o ConnectTimeout=12 -p 31103 \
    a25689@10.239.121.22 'echo H100_OK; ls /home/a25689/BEHAVIOR-2026/datasets 2>/dev/null | head; du -sh /home/a25689/BEHAVIOR-2026/datasets /home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K 2>/dev/null' \
    >>"$REPORT" 2>&1 || echo "H100_SSH_FAIL" >>"$REPORT"
else
  echo "NO_SSHPASS" >>"$REPORT"
  # try bare ssh (may fail without key)
  ssh -o BatchMode=yes -o ConnectTimeout=8 -p 31103 a25689@10.239.121.22 'echo H100_KEY_OK' >>"$REPORT" 2>&1 || echo "NO_KEY" >>"$REPORT"
fi

# disk reclaim candidates (own only)
{
  echo "--- own large dirs ---"
  du -sh /home/a26125/vista_bench /home/a26125/_carla_content_ue5 /home/a26125/.cache 2>/dev/null || true
  du -sh /home/a26125/miniconda3/pkgs 2>/dev/null || true
  echo "--- fuse ---"
  which sshfs fusermount3 mount.fuse 2>/dev/null || true
  ls /dev/fuse 2>/dev/null || true
  echo "--- conda envs ---"
  ls /home/a26125/miniconda3/envs 2>/dev/null || true
} >>"$REPORT"

cat "$REPORT"
