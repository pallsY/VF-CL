#!/usr/bin/env python3
"""Validation-selected balanced final heads for CIFAR-100 checkpoints."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_output_bias import (
    _load_checkpoint,
    configure_evaluation_determinism,
    final_checkpoint_path,
)
from bic_calibration import fit_final_calibrator
from data_utils import VFLDataset, split_features
from formal_cifar100_metrics import calibration_history, compute_formal_metrics
from models import build_models
from vfl_trainer import VFLTrainer


REGULARIZATION = (0.001, 0.01, 0.1, 1.0)
BLEND_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_classes_from_config(config):
    per_task = int(config["classes_per_task"])
    return {
        task_id: list(range(task_id * per_task, (task_id + 1) * per_task))
        for task_id in range(int(config["num_tasks"]))
    }


def _ordered_classes(task_classes):
    return [
        class_id
        for task_id in sorted(task_classes)
        for class_id in task_classes[task_id]
    ]


def _targets(labels, classes):
    positions = {class_id: position for position, class_id in enumerate(classes)}
    return torch.tensor(
        [positions[int(label)] for label in labels],
        dtype=torch.long,
        device=labels.device,
    )


def summarize_scores(scores, labels, task_classes):
    classes = _ordered_classes(task_classes)
    positions = torch.as_tensor(classes, dtype=torch.long, device=scores.device)
    predictions = positions[scores[:, classes].argmax(dim=1)]
    per_task_accuracy = {}
    task_il = {}
    prediction_fraction = {}
    for task_id, task in sorted(task_classes.items()):
        key = f"task_{task_id}"
        task_tensor = torch.as_tensor(task, dtype=torch.long, device=scores.device)
        mask = torch.isin(labels, task_tensor)
        per_task_accuracy[key] = float(predictions[mask].eq(labels[mask]).float().mean())
        local_predictions = task_tensor[scores[mask][:, task].argmax(dim=1)]
        task_il[key] = float(local_predictions.eq(labels[mask]).float().mean())
        prediction_fraction[key] = float(torch.isin(predictions, task_tensor).float().mean())
    return {
        "overall_accuracy": float(predictions.eq(labels).float().mean()),
        "per_task_accuracy": per_task_accuracy,
        "task_il": task_il,
        "task_prediction_fraction": prediction_fraction,
    }


def _identity_raw_alpha(device):
    return torch.tensor(
        math.log(math.expm1(1.0 - 1e-6)), dtype=torch.float32, device=device
    )


def fit_task_class_bias(logits, labels, task_classes, regularization,
                        steps=600, lr=0.03):
    """Fit task affine parameters plus centered per-class residual biases."""
    classes = _ordered_classes(task_classes)
    task_ids = sorted(task_classes)
    device = logits.device
    task_for_position = []
    for task_position, task_id in enumerate(task_ids):
        task_for_position.extend([task_position] * len(task_classes[task_id]))
    task_for_position = torch.tensor(task_for_position, device=device)
    raw_alpha = torch.nn.Parameter(
        _identity_raw_alpha(device).repeat(len(task_ids))
    )
    beta = torch.nn.Parameter(torch.zeros(len(task_ids), device=device))
    class_bias = torch.nn.Parameter(torch.zeros(len(classes), device=device))
    optimizer = torch.optim.Adam([raw_alpha, beta, class_bias], lr=float(lr))
    targets = _targets(labels, classes)

    def centered_bias():
        output = class_bias.clone()
        for task_position in range(len(task_ids)):
            mask = task_for_position == task_position
            output[mask] = output[mask] - output[mask].mean()
        return output

    def scores():
        alpha = F.softplus(raw_alpha) + 1e-6
        return (
            logits[:, classes] * alpha[task_for_position]
            + beta[task_for_position]
            + centered_bias()
        )

    for _ in range(int(steps)):
        optimizer.zero_grad()
        residual = centered_bias()
        loss = F.cross_entropy(scores(), targets)
        loss = loss + float(regularization) * residual.square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return {
            "raw_alpha": raw_alpha.detach().cpu(),
            "beta": beta.detach().cpu(),
            "class_bias": centered_bias().detach().cpu(),
            "task_for_position": task_for_position.detach().cpu(),
            "classes": classes,
        }


def apply_task_class_bias(logits, state):
    device = logits.device
    raw_alpha = state["raw_alpha"].to(device)
    beta = state["beta"].to(device)
    class_bias = state["class_bias"].to(device)
    task_for_position = state["task_for_position"].to(device)
    classes = state["classes"]
    output = logits.clone()
    output[:, classes] = (
        logits[:, classes] * (F.softplus(raw_alpha) + 1e-6)[task_for_position]
        + beta[task_for_position]
        + class_bias
    )
    return output


def fit_linear_prior(embeddings, labels, initial_weight, initial_bias,
                     regularization, steps=500, lr=0.01):
    """Refit the linear head with an L2 anchor to the continual-learning head."""
    weight0 = initial_weight.detach().to(embeddings.device)
    bias0 = initial_bias.detach().to(embeddings.device)
    weight = torch.nn.Parameter(weight0.clone())
    bias = torch.nn.Parameter(bias0.clone())
    optimizer = torch.optim.Adam([weight, bias], lr=float(lr))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        scores = F.linear(embeddings, weight, bias)
        penalty = (weight - weight0).square().sum(dim=1).mean()
        penalty = penalty + (bias - bias0).square().mean()
        loss = F.cross_entropy(scores, labels) + float(regularization) * penalty
        loss.backward()
        optimizer.step()
    return {"weight": weight.detach().cpu(), "bias": bias.detach().cpu()}


def apply_linear_head(embeddings, state):
    return F.linear(
        embeddings,
        state["weight"].to(embeddings.device),
        state["bias"].to(embeddings.device),
    )


def fit_cosine_centroids(embeddings, labels, classes):
    normalized = F.normalize(embeddings, dim=1)
    centroids = []
    for class_id in classes:
        values = normalized[labels == int(class_id)]
        if values.numel() == 0:
            raise ValueError(f"class {class_id} has no calibration embeddings")
        centroids.append(F.normalize(values.mean(dim=0), dim=0))
    return torch.stack(centroids).detach().cpu()


def cosine_scores(embeddings, centroids):
    return F.normalize(embeddings, dim=1) @ centroids.to(embeddings.device).t()


def _row_standardize(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True
    ).clamp(min=1e-6)


def replace_final_metrics(results, final_summary):
    history = calibration_history(results.get("bic_history") or [])
    if len(history) != 10:
        raise ValueError("expected ten legal calibration events")
    history[-1] = {
        "step": history[-1]["step"],
        "per_task_accs": final_summary["per_task_accuracy"],
        "per_task_accs_taskil": final_summary["task_il"],
    }
    return compute_formal_metrics(history, expected_tasks=10)


@torch.no_grad()
def collect_embeddings_logits(trainer, loader, args):
    modules = [*trainer.bottoms, trainer.top_model]
    for module in modules:
        module.eval()
    embeddings, logits, labels = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(args.device)
        parts = split_features(batch_x, args)
        party_embeddings = [
            trainer.bottoms[index](parts[index])
            for index in range(args.num_parties)
        ]
        aggregated = trainer._aggregate(party_embeddings)
        embeddings.append(aggregated.cpu())
        logits.append(trainer.top_model(aggregated).cpu())
        labels.append(batch_y.cpu())
    return torch.cat(embeddings), torch.cat(logits), torch.cat(labels)


def _load_run(run_dir, output_dir, evaluation_source, device):
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "config.json")
    configure_evaluation_determinism(int(config["seed"]))
    values = dict(config)
    values.update(output_dir=str(output_dir), device=device, num_workers=0)
    args = argparse.Namespace(**values)
    dataset = VFLDataset(args)

    saved_calibration = _read_json(run_dir / "bic" / "calibration_manifest.json")
    if dataset.calibration_manifest != saved_calibration:
        raise ValueError("calibration manifest replay mismatch")
    if evaluation_source == "validation":
        saved_validation = _read_json(
            run_dir / "validation" / "validation_manifest.json"
        )
        if dataset.validation_manifest != saved_validation:
            raise ValueError("validation manifest replay mismatch")
        selection_audit = dataset.selection_audit()
        if not selection_audit["passed"] or selection_audit["test_used_for_selection"]:
            raise ValueError("validation selection audit failed")
    elif evaluation_source != "test":
        raise ValueError("evaluation_source must be validation or test")

    checkpoint_path = final_checkpoint_path(str(run_dir))
    checkpoint = _load_checkpoint(checkpoint_path)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint["trainer_state"])
    classes = list(range(int(config["num_classes"])))
    calibration_loader = dataset.get_calibration_loader(classes)
    if evaluation_source == "validation":
        evaluation_loader = dataset.get_validation_loader(classes)
    else:
        _, evaluation_loader = dataset.get_task_loaders(
            classes, shuffle_train=False
        )
    calibration = collect_embeddings_logits(trainer, calibration_loader, args)
    evaluation = collect_embeddings_logits(trainer, evaluation_loader, args)
    return {
        "config": config,
        "results": _read_json(run_dir / "results.json"),
        "task_classes": task_classes_from_config(config),
        "calibration": calibration,
        "evaluation": evaluation,
        "initial_weight": trainer.top_model.classifier.weight.detach().cpu(),
        "initial_bias": trainer.top_model.classifier.bias.detach().cpu(),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "selection_audit": (
            dataset.selection_audit() if evaluation_source == "validation" else None
        ),
        "calibration_audit": dataset.calibration_audit(),
    }


def _task_affine_scores(calibration_logits, calibration_labels,
                        evaluation_logits, task_classes):
    calibrator, fit = fit_final_calibrator(
        calibration_logits, calibration_labels, task_classes,
        "joint_alpha_beta", lr=0.05, steps=1000,
    )
    return calibrator.apply(evaluation_logits), fit


def evaluate_candidates(run_dir, output_dir, output_json, device="cuda:0"):
    loaded = _load_run(run_dir, output_dir, "validation", device)
    cal_emb, cal_logits, cal_labels = [
        value.to(device) for value in loaded["calibration"]
    ]
    val_emb, val_logits, val_labels = [
        value.to(device) for value in loaded["evaluation"]
    ]
    task_classes = loaded["task_classes"]
    classes = _ordered_classes(task_classes)
    records = []

    def add(candidate, scores, fit=None):
        summary = summarize_scores(scores, val_labels, task_classes)
        records.append({
            "candidate": candidate,
            "metrics": replace_final_metrics(loaded["results"], summary),
            "final_summary": summary,
            "fit": fit or {},
        })

    task_scores, task_fit = _task_affine_scores(
        cal_logits, cal_labels, val_logits, task_classes
    )
    add("task_affine", task_scores, task_fit)

    for regularization in REGULARIZATION:
        state = fit_task_class_bias(
            cal_logits, cal_labels, task_classes, regularization
        )
        add(
            f"task_class_bias_reg_{regularization:g}",
            apply_task_class_bias(val_logits, state),
            {"regularization": regularization},
        )

    centroids = fit_cosine_centroids(cal_emb, cal_labels, classes)
    ncm = cosine_scores(val_emb, centroids)
    full_ncm = val_logits.clone()
    full_ncm[:, classes] = ncm
    add("ncm_cosine", full_ncm)
    base = _row_standardize(task_scores[:, classes])
    proto = _row_standardize(ncm)
    for weight in BLEND_WEIGHTS:
        blended = val_logits.clone()
        blended[:, classes] = base + float(weight) * proto
        add(f"ncm_blend_weight_{weight:g}", blended, {"weight": weight})

    initial_weight = loaded["initial_weight"].to(device)
    initial_bias = loaded["initial_bias"].to(device)
    for regularization in REGULARIZATION:
        state = fit_linear_prior(
            cal_emb, cal_labels, initial_weight, initial_bias, regularization
        )
        add(
            f"linear_prior_reg_{regularization:g}",
            apply_linear_head(val_emb, state),
            {"regularization": regularization},
        )

    output = {
        "run_dir": str(Path(run_dir).resolve()),
        "seed": int(loaded["config"]["seed"]),
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "evaluation_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "raw_images_saved": False,
        "party_embeddings_saved": False,
        "calibration_audit": loaded["calibration_audit"],
        "selection_audit": loaded["selection_audit"],
        "records": records,
    }
    _write_json(output_json, output)
    return output


def select_candidate(record_paths, output_json, bwt_floor=-0.20):
    inputs = [_read_json(path) for path in record_paths]
    if {int(item["seed"]) for item in inputs} != {42, 43, 44}:
        raise ValueError("selection requires validation seeds 42, 43, and 44")
    if any(
        item.get("test_used_for_selection")
        or item.get("evaluation_source") != "cifar100-train-validation"
        for item in inputs
    ):
        raise ValueError("selection inputs must be independent validation records")
    by_candidate = {}
    for item in inputs:
        for record in item["records"]:
            by_candidate.setdefault(record["candidate"], []).append(record["metrics"])
    rows = []
    for candidate, metrics in sorted(by_candidate.items()):
        if len(metrics) != 3:
            raise ValueError(f"candidate {candidate} is missing a validation seed")
        row = {"candidate": candidate}
        for metric in ("aa_final_cil", "aa_avg_cil", "bwt_cil", "task_il_final"):
            values = [float(item[metric]) for item in metrics]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values)
        row["eligible"] = row["bwt_cil_mean"] >= float(bwt_floor)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no balanced-head candidate satisfies the BWT floor")
    selected = max(
        eligible,
        key=lambda row: (
            row["aa_final_cil_mean"],
            row["task_il_final_mean"],
            row["bwt_cil_mean"],
            row["candidate"],
        ),
    )
    output = {
        "passed": True,
        "selected_candidate": selected["candidate"],
        "selected_validation_metrics": selected,
        "bwt_floor": float(bwt_floor),
        "test_used_for_selection": False,
        "selection_source": "cifar100-train-validation",
        "validation_records": [str(Path(path).resolve()) for path in record_paths],
        "candidates": rows,
    }
    _write_json(output_json, output)
    return output


def _fit_selected(candidate, loaded, device):
    cal_emb, cal_logits, cal_labels = [
        value.to(device) for value in loaded["calibration"]
    ]
    eval_emb, eval_logits, _ = [
        value.to(device) for value in loaded["evaluation"]
    ]
    task_classes = loaded["task_classes"]
    classes = _ordered_classes(task_classes)
    if candidate == "task_affine":
        return _task_affine_scores(
            cal_logits, cal_labels, eval_logits, task_classes
        )[0]
    if candidate.startswith("task_class_bias_reg_"):
        regularization = float(candidate.removeprefix("task_class_bias_reg_"))
        state = fit_task_class_bias(
            cal_logits, cal_labels, task_classes, regularization
        )
        return apply_task_class_bias(eval_logits, state)
    if candidate == "ncm_cosine" or candidate.startswith("ncm_blend_weight_"):
        centroids = fit_cosine_centroids(cal_emb, cal_labels, classes)
        ncm = cosine_scores(eval_emb, centroids)
        output = eval_logits.clone()
        if candidate == "ncm_cosine":
            output[:, classes] = ncm
            return output
        weight = float(candidate.removeprefix("ncm_blend_weight_"))
        task_scores = _task_affine_scores(
            cal_logits, cal_labels, eval_logits, task_classes
        )[0]
        output[:, classes] = (
            _row_standardize(task_scores[:, classes])
            + weight * _row_standardize(ncm)
        )
        return output
    if candidate.startswith("linear_prior_reg_"):
        regularization = float(candidate.removeprefix("linear_prior_reg_"))
        state = fit_linear_prior(
            cal_emb, cal_labels,
            loaded["initial_weight"].to(device),
            loaded["initial_bias"].to(device),
            regularization,
        )
        return apply_linear_head(eval_emb, state)
    raise ValueError(f"unknown selected candidate: {candidate}")


def evaluate_formal(run_dir, output_dir, selection_path, output_json,
                    device="cuda:0"):
    selection = _read_json(selection_path)
    if not selection.get("passed") or selection.get("test_used_for_selection"):
        raise ValueError("formal evaluation requires a legal frozen selection")
    candidate = selection["selected_candidate"]
    loaded = _load_run(run_dir, output_dir, "test", device)
    scores = _fit_selected(candidate, loaded, device)
    labels = loaded["evaluation"][2].to(device)
    summary = summarize_scores(scores, labels, loaded["task_classes"])
    output = {
        "run_dir": str(Path(run_dir).resolve()),
        "seed": int(loaded["config"]["seed"]),
        "candidate": candidate,
        "metrics": replace_final_metrics(loaded["results"], summary),
        "final_summary": summary,
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "selection_sha256": _sha256(selection_path),
        "selection_test_used": False,
        "test_used_for_fit": False,
        "raw_images_saved": False,
        "party_embeddings_saved": False,
        "calibration_audit": loaded["calibration_audit"],
    }
    _write_json(output_json, output)
    return output


def summarize_formal(record_paths, selection_path, output_json):
    records = [_read_json(path) for path in record_paths]
    if {int(item["seed"]) for item in records} != {42, 43, 44}:
        raise ValueError("formal summary requires seeds 42, 43, and 44")
    selection_sha = _sha256(selection_path)
    candidate = _read_json(selection_path)["selected_candidate"]
    if any(
        item["candidate"] != candidate
        or item["selection_sha256"] != selection_sha
        or item["selection_test_used"]
        or item["test_used_for_fit"]
        for item in records
    ):
        raise ValueError("formal records do not share the frozen legal selection")
    summary = {"candidate": candidate}
    for metric in ("aa_final_cil", "aa_avg_cil", "bwt_cil", "task_il_final"):
        values = [float(item["metrics"][metric]) for item in records]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    output = {
        "passed": True,
        "selection_sha256": selection_sha,
        "selection_test_used": False,
        "test_used_for_fit": False,
        "records": [str(Path(path).resolve()) for path in record_paths],
        "summary": summary,
    }
    _write_json(output_json, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--run-dir", required=True)
    validation.add_argument("--output-dir", required=True)
    validation.add_argument("--output-json", required=True)
    validation.add_argument("--device", default="cuda:0")

    select = subparsers.add_parser("select")
    select.add_argument("--records", nargs=3, required=True)
    select.add_argument("--output-json", required=True)
    select.add_argument("--bwt-floor", type=float, default=-0.20)

    formal = subparsers.add_parser("formal")
    formal.add_argument("--run-dir", required=True)
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--selection", required=True)
    formal.add_argument("--output-json", required=True)
    formal.add_argument("--device", default="cuda:0")

    aggregate = subparsers.add_parser("summarize")
    aggregate.add_argument("--records", nargs=3, required=True)
    aggregate.add_argument("--selection", required=True)
    aggregate.add_argument("--output-json", required=True)

    args = parser.parse_args()
    if args.command == "validation":
        output = evaluate_candidates(
            args.run_dir, args.output_dir, args.output_json, args.device
        )
    elif args.command == "select":
        output = select_candidate(
            args.records, args.output_json, args.bwt_floor
        )
    elif args.command == "formal":
        output = evaluate_formal(
            args.run_dir, args.output_dir, args.selection,
            args.output_json, args.device,
        )
    else:
        output = summarize_formal(
            args.records, args.selection, args.output_json
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
