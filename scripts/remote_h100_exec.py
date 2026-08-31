#!/usr/bin/env python3
"""Password SSH helpers for H100 jump hosts used by this project."""
from __future__ import annotations

import argparse
import shlex
import sys

import pexpect

HOSTS = {
    "22": {
        "host": "a25689@10.239.121.22",
        "password": "123456",
        "port": "31103",
    },
    "21": {
        "host": "a25689@10.239.121.21",
        "password": "123456",
        "port": "31126",
    },
}


def run(which: str, remote_cmd: str, timeout: int = 300) -> int:
    conf = HOSTS[which]
    wrapped = "bash --noprofile --norc -c " + shlex.quote(remote_cmd)
    argv = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-p", conf["port"],
        conf["host"],
        wrapped,
    ]
    child = pexpect.spawn(argv[0], argv[1:], encoding="utf-8", timeout=timeout)
    child.logfile_read = sys.stdout
    i = child.expect(["password:", "(?i)are you sure you want to continue connecting", pexpect.EOF, pexpect.TIMEOUT])
    if i == 1:
        child.sendline("yes")
        child.expect("password:")
        child.sendline(conf["password"])
    elif i == 0:
        child.sendline(conf["password"])
    else:
        raise RuntimeError(f"login failed idx={i}")
    child.expect(pexpect.EOF)
    child.close()
    return int(child.exitstatus or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("host", choices=sorted(HOSTS))
    ap.add_argument("remote_cmd")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    return run(args.host, args.remote_cmd, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
