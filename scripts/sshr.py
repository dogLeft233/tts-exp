#!/usr/bin/env python3
"""SSH helper: run a remote command or copy files using env-carried credentials.

Usage:
  sshr.py run 'remote command'                # execute and stream output
  sshr.py put <local> <remote>                # upload a file
  sshr.py get <remote> <local>                # download a file
  sshr.py put-dir <local> <remote>            # upload a directory (recursive)
  sshr.py get-dir <remote> <local>            # download a directory (recursive)

Credentials come from env: TTS_SSH_HOST, TTS_SSH_PORT, TTS_SSH_USER, TTS_SSH_PASS.
Passwords never appear in files or command lines.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("TTS_SSH_HOST", "connect.westc.seetacloud.com")
PORT = int(os.environ.get("TTS_SSH_PORT", "13201"))
USER = os.environ.get("TTS_SSH_USER", "root")
PASSWORD = os.environ.get("TTS_SSH_PASS")
if not PASSWORD:
    sys.exit("TTS_SSH_PASS is required (export the password, never write it to a file)")


def client() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=25)
    return ssh


def run(command: str) -> int:
    ssh = client()
    _, stdout, stderr = ssh.exec_command(command, timeout=3600, get_pty=False)
    for line in stdout:
        print(line, end="")
    for line in stderr:
        print(line, end="", file=sys.stderr)
    code = stdout.channel.recv_exit_status()
    ssh.close()
    return code


def put(local: str, remote: str) -> None:
    ssh = client()
    sftp = ssh.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    ssh.close()
    print(f"uploaded {local} -> {remote}")


def get(remote: str, local: str) -> None:
    ssh = client()
    sftp = ssh.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    ssh.close()
    print(f"downloaded {remote} -> {local}")


def _sftp_mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.split("/")
    current = parts[0] if parts[0] else "/"
    for part in parts[1:]:
        current = f"{current}/{part}" if current != "/" else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def put_dir(local: str, remote: str) -> None:
    ssh = client()
    sftp = ssh.open_sftp()
    _sftp_mkdirs(sftp, remote)
    local_root = Path(local)

    def walk(src: Path, dst: str) -> None:
        for entry in sorted(src.iterdir()):
            target = f"{dst}/{entry.name}"
            if entry.is_dir():
                _sftp_mkdirs(sftp, target)
                walk(entry, target)
            else:
                sftp.put(str(entry), target)
                print(f"uploaded {entry} -> {target}")

    walk(local_root, remote)
    sftp.close()
    ssh.close()


def get_dir(remote: str, local: str) -> None:
    ssh = client()
    sftp = ssh.open_sftp()
    local_root = Path(local)
    local_root.mkdir(parents=True, exist_ok=True)

    def walk(dst: str, src: Path) -> None:
        for entry in sftp.listdir_attr(dst):
            remote_path = f"{dst}/{entry.filename}"
            local_path = src / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                local_path.mkdir(exist_ok=True)
                walk(remote_path, local_path)
            else:
                sftp.get(remote_path, str(local_path))
                print(f"downloaded {remote_path} -> {local_path}")

    walk(remote, local_root)
    sftp.close()
    ssh.close()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    action = sys.argv[1]
    if action == "run":
        return run(sys.argv[2])
    if action == "put":
        put(sys.argv[2], sys.argv[3])
        return 0
    if action == "get":
        get(sys.argv[2], sys.argv[3])
        return 0
    if action == "put-dir":
        put_dir(sys.argv[2], sys.argv[3])
        return 0
    if action == "get-dir":
        get_dir(sys.argv[2], sys.argv[3])
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
