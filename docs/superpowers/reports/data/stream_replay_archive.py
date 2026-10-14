"""Stream a completed remote run into a local archive without remote tar space."""

import argparse
import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True, choices=("vf-cl-4090-1", "vf-cl-3080", "vf-cl-5090-1"))
parser.add_argument("--remote-root", required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--destination", required=True, type=Path)
args = parser.parse_args()

if not args.remote_root.startswith("/home/") or ".." in args.remote_root.split("/"):
    raise ValueError("remote root must be an absolute home result path")
if not re.fullmatch(r"[A-Za-z0-9_]+", args.run_name):
    raise ValueError("run name must be a single directory name")
if args.destination.suffixes[-2:] != [".tar", ".gz"]:
    raise ValueError("destination must end in .tar.gz")

partial = args.destination.with_name(args.destination.name + ".partial")
if args.destination.exists() or partial.exists():
    raise FileExistsError("destination or partial archive already exists")
remote = (f"tar -C {shlex.quote(args.remote_root)} -czf - "
          f"{shlex.quote(args.run_name)}")
digest = hashlib.sha256()
size = 0
process = subprocess.Popen(
    ["ssh", "-o", "BatchMode=yes", args.host, remote],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
try:
    with partial.open("xb") as output:
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            output.write(block)
            digest.update(block)
            size += len(block)
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"remote tar failed; partial archive retained: {error}")
finally:
    if process.poll() is None:
        process.terminate()
        process.wait()

os.replace(partial, args.destination)
print(f"STREAMED bytes={size} sha256={digest.hexdigest()} path={args.destination}")
