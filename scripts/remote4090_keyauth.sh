#!/usr/bin/env bash
set -uo pipefail
KEY=/home/a26125/.ssh/id_ed25519_behavior
OUT=/home/a26125/behavior_logs/keyauth.txt
{
  echo "=== $(date -Is) ==="
  echo "pub:"; cat "${KEY}.pub"
  echo "--- port 31103 ---"
  ssh -vvv -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no -o BatchMode=yes -o ConnectTimeout=8 \
    -p 31103 a25689@10.239.121.22 'echo KEY_OK_31103' 2>&1 | tail -50
  echo "exit_31103:$?"
  echo "--- port 22 ---"
  ssh -vvv -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no -o BatchMode=yes -o ConnectTimeout=8 \
    -p 22 a25689@10.239.121.22 'echo KEY_OK_22' 2>&1 | tail -50
  echo "exit_22:$?"
} | tee "$OUT"
