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
assert actual_archive == args.archive_sha256, "archive SHA-256 mismatch"

expected = {}
with open(args.manifest, encoding="utf-8") as stream:
    for line in stream:
        value, name = line.rstrip("\n").split("  ", 1)
        assert name not in expected, "duplicate path in manifest"
        expected[name] = value

with tarfile.open(args.archive, "r:gz") as archive:
    members = [member for member in archive if member.isfile()]
    assert len(members) == len(expected), "file count mismatch"
    assert {member.name for member in members} == set(expected), "file path mismatch"
    for member in members:
        with archive.extractfile(member) as stream:
            assert digest(stream) == expected[member.name], member.name

print(f"VERIFIED archive={actual_archive} files={len(expected)}")
