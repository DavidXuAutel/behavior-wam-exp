#!/usr/bin/env bash
set -euo pipefail
KEY=/home/a26125/.ssh/id_ed25519_behavior
ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes \
  -o ConnectTimeout=12 -p 31103 a25689@10.239.121.22 \
  'echo H100_KEY_OK; ls /home/a25689/BEHAVIOR-2026/datasets; du -sh /home/a25689/BEHAVIOR-2026/datasets /home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K 2>/dev/null | head'
