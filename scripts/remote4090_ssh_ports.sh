#!/usr/bin/env bash
set -euo pipefail
KEY=/home/a26125/.ssh/id_ed25519_behavior
LOG=/home/a26125/behavior_logs/ssh_ports.txt
{
  echo "=== $(date -Is) ==="
  for spec in "22:22" "31103:31103" "21:31126"; do
    name=${spec%%:*}; port=${spec##*:}
    host=10.239.121.$name
    if [[ "$name" == "21" ]]; then host=10.239.121.21; fi
    if [[ "$name" == "22" ]]; then host=10.239.121.22; fi
    if [[ "$name" == "31103" ]]; then host=10.239.121.22; port=31103; fi
    echo "--- try $host port $port ---"
    timeout 12 ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
      -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
      -o BatchMode=yes -o ConnectTimeout=8 -p "$port" \
      "a25689@$host" "echo OK_from_${host}_${port}" 2>&1 | tail -20 || echo FAIL_$port
  done
  # also try passwordless default port 22 with password disabled - and password via sshpass if any
  echo "--- password auth port 22 ---"
  if command -v sshpass >/dev/null; then
    sshpass -p 123456 ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no -o ConnectTimeout=8 -p 22 a25689@10.239.121.22 'echo PASS_OK_22' 2>&1 || echo PASS_FAIL_22
  else
    echo NO_SSHPASS
  fi
} | tee "$LOG"
