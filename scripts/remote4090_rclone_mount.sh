#!/usr/bin/env bash
# Configure rclone SFTP mount of H100 BEHAVIOR datasets on 4090
set -euo pipefail
RCLONE=/home/a26125/behavior_logs/rclone
KEY=/home/a26125/.ssh/id_ed25519_behavior
MNT=/home/a26125/behavior_mnt
mkdir -p "$MNT" /home/a26125/.config/rclone

"$RCLONE" config create h100sftp sftp \
  host 10.239.121.22 \
  user a25689 \
  port 31103 \
  key_file "$KEY" \
  shell_type unix \
  md5sum_command none \
  sha1sum_command none \
  --non-interactive 2>/dev/null || true

# refresh if exists
cat > /home/a26125/.config/rclone/rclone.conf <<EOF
[h100sftp]
type = sftp
host = 10.239.121.22
user = a25689
port = 31103
key_file = $KEY
shell_type = unix
EOF

echo "rclone ls test:"
"$RCLONE" lsd h100sftp:/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K/datasets --max-depth 1 | head

if mountpoint -q "$MNT"; then
  fusermount3 -u "$MNT" || fusermount -u "$MNT" || true
fi

"$RCLONE" mount h100sftp:/home/a25689/BEHAVIOR-2026/code/BEHAVIOR-1K \
  "$MNT" \
  --daemon \
  --vfs-cache-mode full \
  --vfs-cache-max-size 8G \
  --dir-cache-time 10m \
  --log-file /home/a26125/behavior_logs/rclone_mount.log

sleep 2
ls "$MNT" | head
ls "$MNT/datasets" | head
echo MOUNT_OK
