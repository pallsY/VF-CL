#!/usr/bin/env python3
"""Read-only P0 diagnosis for Class-IL output bias."""

import argparse
import glob
import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import VFLDataset, split_features
from models import build_models
from vfl_trainer import VFLTrainer


REQUIRED_DETERMINISTIC_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def deterministic_environment_mismatches(seed, environ=None):
    environ = os.environ if environ is None else environ
    expected = {**REQUIRED_DETERMINISTIC_ENV, "PYTHONHASHSEED": str(seed)}
    return [
        "{}={!r} (expected {!r})".format(key, environ.get(key), value)
        for key, value in expected.items()
        if environ.get(key) != value
    ]


def configure_evaluation_determinism(seed):
    mismatches = deterministic_environment_mismatches(seed)
    if mismatches:
        raise RuntimeError(
            "deterministic environment contract failed: " + "; ".join(mismatches)
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_sum_decomposition(full_logits, party_logits, classifier_bias, atol=1e-5):
    full = np.asarray(full_logits, dtype=np.float64)
    party = np.asarray(party_logits, dtype=np.float64)
    bias = np.asarray(classifier_bias, dtype=np.float64)
    rebuilt = party.sum(axis=1) + bias[None, :]
    if full.shape != rebuilt.shape:
        raise ValueError("decomposition shape mismatch")
    error = float(np.max(np.abs(full - rebuilt)))
    if error > atol:
        raise ValueError("decomposition mismatch: max_abs_error={:.8g}".format(error))
    return error


def task_parameter_summary(weight, classifier_bias, classes_per_task):
    row_norm = np.linalg.norm(np.asarray(weight, dtype=np.float64), axis=1)
    bias = np.asarray(classifier_bias, dtype=np.float64)
    if row_norm.size != bias.size or row_norm.size % classes_per_task:
        raise ValueError("classifier rows are incompatible with classes_per_task")
    return {
        "mean_row_norm": row_norm.reshape(-1, classes_per_task).mean(axis=1).tolist(),
        "mean_classifier_bias": bias.reshape(-1, classes_per_task).mean(axis=1).tolist(),
        "row_norm_by_class": row_norm.tolist(),
        "classifier_bias_by_class": bias.tolist(),
    }


def _validate_logit_shapes(labels, full_logits, party_logits, classes_per_task):
    labels = np.asarray(labels, dtype=np.int64)
    full = np.asarray(full_logits, dtype=np.float64)
    party = np.asarray(party_logits, dtype=np.float64)
    if full.ndim != 2 or party.ndim != 3 or full.shape[0] != labels.size:
        raise ValueError("incompatible label/logit dimensions")
    if party.shape[0] != labels.size or party.shape[2] != full.shape[1]:
        raise ValueError("party logits have incompatible shape")
    if full.shape[1] % classes_per_task:
        raise ValueError("class count is incompatible with classes_per_task")
    if labels.size == 0:
        raise ValueError("empty evaluation data")
    return labels, full, party


def task_logit_summary(labels, full_logits, party_logits, classes_per_task):
    labels, full, party = _validate_logit_shapes(
        labels, full_logits, party_logits, classes_per_task
    )
    num_tasks = full.shape[1] // classes_per_task
    task_logits = full.reshape(len(labels), num_tasks, classes_per_task).mean(axis=2)
    party_task_logits = party.reshape(
        len(labels), party.shape[1], num_tasks, classes_per_task
    ).mean(axis=3)
    true_tasks = labels // classes_per_task
    if true_tasks.max() >= num_tasks:
        raise ValueError("labels exceed classifier task range")
    mean_task_logits = []
    mean_party_logits = []
    mean_party_by_class_task = []
    for task_id in range(num_tasks):
        mask = true_tasks == task_id
        if not np.any(mask):
            raise ValueError("missing examples for task {}".format(task_id))
        mean_task_logits.append(task_logits[mask].mean(axis=0).tolist())
        mean_party_logits.append(party[mask].mean(axis=(0, 2)).tolist())
        mean_party_by_class_task.append(
            party_task_logits[mask].mean(axis=0).tolist()
        )
    return {
        "mean_task_logit_by_true_task": mean_task_logits,
        "mean_party_logit_by_true_task": mean_party_logits,
        "mean_party_logit_by_true_and_class_task": mean_party_by_class_task,
    }


def task_probability_summary(labels, logits, classes_per_task):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    num_tasks = logits.shape[1] // classes_per_task
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    task_mass = probabilities.reshape(
        len(labels), num_tasks, classes_per_task
    ).sum(axis=2)
    true_tasks = labels // classes_per_task
    predicted_tasks = logits.argmax(axis=1) // classes_per_task
    return {
        "mean_task_probability_mass_by_true_task": [
            task_mass[true_tasks == task_id].mean(axis=0).tolist()
            for task_id in range(num_tasks)
        ],
        "predicted_task_fraction": (
            np.bincount(predicted_tasks, minlength=num_tasks) / len(labels)
        ).tolist(),
    }


def validate_run_contract(config):
    required = {
        "deterministic": 1,
        "data": "cifar100",
        "num_tasks": 10,
        "classes_per_task": 10,
        "num_parties": 4,
        "model_type": "resnet18",
        "aggregation": "sum",
        "cosine_head": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                "protocol mismatch for {}: {!r}".format(key, config.get(key))
            )


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_run(pilot_root, variant, seed):
    matches = []
    for result_path in glob.glob(os.path.join(pilot_root, "*", "results.json")):
        run_dir = os.path.dirname(result_path)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if (
            config.get("expected_party_kd_variant") == variant
            and int(config.get("seed")) == seed
        ):
            matches.append((run_dir, config))
    if len(matches) != 1:
        raise RuntimeError(
            "expected one {} seed {} run, found {}".format(variant, seed, len(matches))
        )
    return matches[0]


def final_checkpoint_path(run_dir):
    path = os.path.join(run_dir, "checkpoints", "event_9_CIL.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "full CIL checkpoint is required; audit-only files are insufficient: {}".format(
                path
            )
        )
    return path


def _evaluation_args(config, data_path):
    values = dict(config)
    values["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    values["data_path"] = data_path or config["data_path"]
    return argparse.Namespace(**values)


@torch.no_grad()
def replay_final_checkpoint(args, checkpoint):
    dataset = VFLDataset(args)
    _, test_loader = dataset.get_task_loaders(list(range(args.num_classes)))
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint["trainer_state"])
    for module in [*trainer.bottoms, trainer.top_model]:
        module.eval()

    all_labels, all_logits, all_party_logits = [], [], []
    classifier_weight = trainer.top_model.classifier.weight
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(args.device)
        parts = split_features(batch_x, args)
        embeddings = [trainer.bottoms[p](parts[p]) for p in range(args.num_parties)]
        full_logits = trainer.top_model(sum(embeddings))
        party_logits = torch.stack(
            [F.linear(embeddings[p], classifier_weight, None) for p in range(args.num_parties)],
            dim=1,
        )
        all_labels.append(batch_y.cpu())
        all_logits.append(full_logits.cpu())
        all_party_logits.append(party_logits.cpu())

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_logits).numpy(),
        torch.cat(all_party_logits).numpy(),
        trainer.top_model.classifier.weight.detach().cpu().numpy(),
        trainer.top_model.classifier.bias.detach().cpu().numpy(),
    )


def _analyze_run_dir(run_dir, data_path, variant=None, seed=None):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    validate_run_contract(config)
    variant = variant or config.get("expected_party_kd_variant")
    seed = int(config["seed"] if seed is None else seed)
    if config.get("expected_party_kd_variant") != variant:
        raise ValueError("variant contract mismatch")
    if int(config["seed"]) != seed:
        raise ValueError("seed contract mismatch")
    configure_evaluation_determinism(seed)
    checkpoint_path = final_checkpoint_path(run_dir)
    checkpoint = _load_checkpoint(checkpoint_path)
    labels, logits, party_logits, weight, bias = replay_final_checkpoint(
        _evaluation_args(config, data_path), checkpoint
    )
    decomposition_error = validate_sum_decomposition(logits, party_logits, bias)
    replayed_accuracy = float(np.mean(logits.argmax(axis=1) == labels))
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as handle:
        stored_accuracy = float(json.load(handle)["cl_metrics"]["AA_final"])
    arrays = np.load(os.path.join(run_dir, "final_probs.npz"))
    saved_accuracy = float(np.mean(arrays["probs"].argmax(axis=1) == arrays["labels"]))
    if abs(replayed_accuracy - stored_accuracy) > 1e-12:
        raise ValueError("replayed AA_final does not match results.json")
    if abs(replayed_accuracy - saved_accuracy) > 1e-12:
        raise ValueError("replayed AA_final does not match final_probs.npz")
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_contract": {
            key: config[key]
            for key in (
                "deterministic", "data", "num_tasks", "classes_per_task",
                "num_parties", "model_type", "aggregation", "cosine_head",
                "expected_party_kd_variant",
            )
        },
        "stored_aa_final": stored_accuracy,
        "replayed_aa_final": replayed_accuracy,
        "saved_probs_aa_final": saved_accuracy,
        "decomposition_max_abs_error": decomposition_error,
        "decomposition_atol": 1e-5,
        "classifier_parameters": task_parameter_summary(
            weight, bias, config["classes_per_task"]
        ),
        "task_probability": task_probability_summary(
            labels, logits, config["classes_per_task"]
        ),
        "task_logits": task_logit_summary(
            labels, logits, party_logits, config["classes_per_task"]
        ),
    }


def _analyze_run(pilot_root, variant, seed, data_path):
    run_dir, _ = _find_run(pilot_root, variant, seed)
    return _analyze_run_dir(run_dir, data_path, variant=variant, seed=seed)


def _write_summary(path, records):
    lines = [
        "# P0 Output-Bias Source Diagnosis",
        "",
        "Read-only checkpoint replay; no correction is fitted.",
        "",
        "| Variant | Seed | AA_final | Task-9 predicted fraction | Mean norm: task 0 | Mean norm: task 9 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        norms = record["classifier_parameters"]["mean_row_norm"]
        newest_fraction = record["task_probability"]["predicted_task_fraction"][-1]
        lines.append(
            "| {variant} | {seed} | {accuracy:.4f} | {fraction:.4f} | {old:.4f} | {new:.4f} |".format(
                variant=record["variant"],
                seed=record["seed"],
                accuracy=record["replayed_aa_final"],
                fraction=newest_fraction,
                old=norms[0],
                new=norms[-1],
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Analyse one run directory containing checkpoints/event_9_CIL.pt.",
    )
    parser.add_argument("--variants", nargs="+", default=["static", "uniform"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()

    if args.run_dir:
        records = [_analyze_run_dir(args.run_dir, args.data_path)]
    else:
        records = [
            _analyze_run(args.pilot_root, variant, seed, args.data_path)
            for variant in args.variants
            for seed in args.seeds
        ]
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "protocol": {
            "diagnostic_only": True,
            "party_logit_formula": "F.linear(embedding_p, classifier_weight, None)",
            "reconstruction": "sum_party_logits + classifier_bias",
        },
        "records": records,
    }
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    _write_summary(os.path.join(args.output_dir, "SUMMARY.md"), records)
    print(os.path.join(args.output_dir, "SUMMARY.md"))


if __name__ == "__main__":
    main()
