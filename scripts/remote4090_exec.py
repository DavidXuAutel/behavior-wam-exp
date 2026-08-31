#!/usr/bin/env python3
"""Password SSH helper for a26125@10.229.20.110"""
from __future__ import annotations

import argparse
import base64
import shlex
import sys

import pexpect

HOST = "a26125@10.229.20.110"
PASSWORD = "Autel123"
SSH_BASE = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "NumberOfPasswordPrompts=1",
]


def _login(argv: list[str], timeout: int = 120) -> pexpect.spawn:
    child = pexpect.spawn(
        argv[0], argv[1:], encoding="utf-8", timeout=timeout, codec_errors="replace"
    )
    i = child.expect(
        ["password:", "(?i)are you sure you want to continue connecting", pexpect.EOF, pexpect.TIMEOUT]
    )
    if i == 1:
        child.sendline("yes")
        child.expect("password:")
        child.sendline(PASSWORD)
    elif i == 0:
        child.sendline(PASSWORD)
    else:
        raise RuntimeError(f"ssh login failed idx={i} before={child.before!r}")
    return child


def run(remote_cmd: str, timeout: int = 300) -> int:
    # Critical: pass ONE remote argv so ssh does not split `bash -c` script.
    wrapped = "bash --noprofile --norc -c " + shlex.quote(remote_cmd)
    child = _login(SSH_BASE + [HOST, wrapped], timeout=timeout)
    child.logfile_read = sys.stdout
    child.expect(pexpect.EOF)
    child.close()
    return int(child.exitstatus or 0)


def put(local: str, remote: str, timeout: int = 300) -> int:
    with open(local, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    remote_cmd = "\n".join(
        [
            "python3 - <<'PY'",
            "import base64, os",
            f"p = {remote!r}",
            f"open(p, 'wb').write(base64.b64decode({b64!r}))",
            "os.chmod(p, 0o755)",
            "print('WROTE', len(open(p, 'rb').read()))",
            "PY",
        ]
    )
    return run(remote_cmd, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("remote_cmd")
    r.add_argument("--timeout", type=int, default=300)
    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")
    args = ap.parse_args()
    if args.cmd == "run":
        return run(args.remote_cmd, timeout=args.timeout)
    return put(args.local, args.remote)


if __name__ == "__main__":
    raise SystemExit(main())
