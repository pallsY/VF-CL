"""Verify every regular file in a saved training run against its SHA-256 list."""

import argparse
import hashlib
import tarfile


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("archive")
parser.add_argument("manifest")
parser.add_argument("--archive-sha256", required=True)
args = parser.parse_args()

with open(args.archive, "rb") as stream:
    actual_archive = digest(stream)
if actual_archive != args.archive_sha256:
    raise ValueError("archive SHA-256 mismatch")

expected = {}
with open(args.manifest, encoding="utf-8") as stream:
    for line in stream:
        value, name = line.rstrip("\n").split("  ", 1)
        if name in expected:
            raise ValueError("duplicate path in manifest")
        expected[name] = value

seen = set()
with tarfile.open(args.archive, "r|gz") as archive:
    for member in archive:
        if member.isdir():
            continue
        if not member.isfile() or member.name not in expected or member.name in seen:
            raise ValueError(f"unexpected archive member: {member.name}")
        with archive.extractfile(member) as stream:
            if digest(stream) != expected[member.name]:
                raise ValueError(f"file SHA-256 mismatch: {member.name}")
        seen.add(member.name)
if seen != set(expected):
    raise ValueError("archive file count or path mismatch")

print(f"VERIFIED archive={actual_archive} files={len(expected)}")
