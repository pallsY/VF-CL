import hashlib
import os

import torch


FORMAT_VERSION = 1


def tensor_sha256(weights):
    weights = weights.detach().cpu().contiguous()
    header = f"{weights.dtype}|{tuple(weights.shape)}|".encode("ascii")
    return hashlib.sha256(header + weights.numpy().tobytes()).hexdigest()


def build_task_derangement(classes, seed):
    classes = [int(c) for c in classes]
    if len(classes) < 2:
        raise ValueError("a derangement requires at least two classes")
    if len(set(classes)) != len(classes):
        raise ValueError("task classes must be unique")
    generator = torch.Generator().manual_seed(int(seed))
    identity = torch.arange(len(classes))
    while True:
        permutation = torch.randperm(len(classes), generator=generator)
        if torch.all(permutation != identity):
            return {target: classes[int(permutation[i])] for i, target in enumerate(classes)}


def build_manifest(weights, training_seed, task_classes, source_run, source_checkpoint):
    weights = weights.detach().cpu().contiguous()
    if weights.ndim != 2:
        raise ValueError(f"weights must be [classes, parties], got {tuple(weights.shape)}")
    if weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32, got {weights.dtype}")
    flat_classes = [int(c) for task in task_classes for c in task]
    if sorted(flat_classes) != list(range(weights.shape[0])):
        raise ValueError("task_classes must partition every manifest class exactly once")
    weights = weights.clone()
    return {
        "format_version": FORMAT_VERSION,
        "training_seed": int(training_seed),
        "num_classes": int(weights.shape[0]),
        "num_parties": int(weights.shape[1]),
        "task_classes": [[int(c) for c in task] for task in task_classes],
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "weights": weights,
        "tensor_sha256": tensor_sha256(weights),
    }


def save_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(manifest, path)


def load_manifest(path, expected_seed=None, expected_parties=None, expected_classes=None):
    manifest = torch.load(path, map_location="cpu", weights_only=True)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported manifest format: {manifest.get('format_version')}")
    weights = manifest.get("weights")
    if not isinstance(weights, torch.Tensor) or weights.dtype != torch.float32 or weights.ndim != 2:
        raise ValueError("manifest weights must be a 2D float32 tensor")
    if tensor_sha256(weights) != manifest.get("tensor_sha256"):
        raise ValueError("manifest tensor hash mismatch")
    if expected_seed is not None and manifest.get("training_seed") != int(expected_seed):
        raise ValueError("manifest training seed mismatch")
    if expected_parties is not None and weights.shape[1] != int(expected_parties):
        raise ValueError("manifest party count mismatch")
    if expected_classes is not None and weights.shape[0] != int(expected_classes):
        raise ValueError("manifest class count mismatch")
    return manifest
