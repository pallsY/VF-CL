#!/usr/bin/env python3
"""Stage C validation-selected hierarchical heads for CIFAR-100."""
import argparse
import json
import re
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from offline_balanced_head import (
    _identity_raw_alpha,
    _load_run,
    _ordered_classes,
    _sha256,
    _targets,
    _task_affine_scores,
    _write_json,
    apply_task_class_bias,
    replace_final_metrics,
    select_candidate,
    summarize_scores,
)


CLASS_REGULARIZATION = (0.0003, 0.001, 0.003, 0.01)
TASK_REGULARIZATION = (0.0, 0.0001, 0.001, 0.01)
TASK_WEIGHTS = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
NAME_PATTERN = re.compile(
    r"^tcb_c(?P<class_reg>[0-9.e-]+)_t(?P<task_reg>[0-9.e-]+)"
    r"_g(?P<task_weight>[0-9.]+)$"
)


def candidate_name(class_reg, task_reg, task_weight):
    return (
        f"tcb_c{class_reg:g}_t{task_reg:g}_g{task_weight:g}"
    )


def candidate_specs():
    return [
        (class_reg, task_reg, task_weight)
        for class_reg in CLASS_REGULARIZATION
        for task_reg in TASK_REGULARIZATION
        for task_weight in TASK_WEIGHTS
    ]


def parse_candidate(candidate):
    match = NAME_PATTERN.fullmatch(candidate)
    if not match:
        raise ValueError(f"unknown Stage C candidate: {candidate}")
    return tuple(float(match.group(key)) for key in (
        "class_reg", "task_reg", "task_weight"
    ))


def fit_regularized_task_class_bias(
        logits, labels, task_classes, class_regularization,
        task_regularization, steps=600, lr=0.03):
    """Fit task affine and centered class bias with identity anchoring."""
    classes = _ordered_classes(task_classes)
    task_ids = sorted(task_classes)
    device = logits.device
    task_for_position = []
    for task_position, task_id in enumerate(task_ids):
        task_for_position.extend(
            [task_position] * len(task_classes[task_id])
        )
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

    def parameters():
        alpha = F.softplus(raw_alpha) + 1e-6
        centered_beta = beta - beta.mean()
        return alpha, centered_beta, centered_bias()

    def scores():
        alpha, centered_beta, residual = parameters()
        return (
            logits[:, classes] * alpha[task_for_position]
            + centered_beta[task_for_position]
            + residual
        )

    for _ in range(int(steps)):
        optimizer.zero_grad()
        alpha, centered_beta, residual = parameters()
        loss = F.cross_entropy(scores(), targets)
        loss = loss + float(class_regularization) * residual.square().mean()
        task_penalty = (alpha - 1.0).square().mean()
        task_penalty = task_penalty + centered_beta.square().mean()
        loss = loss + float(task_regularization) * task_penalty
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        alpha, centered_beta, residual = parameters()
        return {
            "raw_alpha": raw_alpha.detach().cpu(),
            "beta": centered_beta.detach().cpu(),
            "class_bias": residual.detach().cpu(),
            "task_for_position": task_for_position.cpu(),
            "classes": classes,
            "fit_summary": {
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "beta_abs_max": float(centered_beta.abs().max()),
                "class_bias_abs_max": float(residual.abs().max()),
            },
        }


def hierarchical_task_scores(scores, task_classes, task_weight):
    """Reweight task evidence while preserving within-task predictions."""
    output = scores.clone()
    evidence = torch.stack([
        torch.logsumexp(scores[:, task_classes[task_id]], dim=1)
        for task_id in sorted(task_classes)
    ], dim=1)
    task_log_probability = F.log_softmax(evidence, dim=1)
    for position, task_id in enumerate(sorted(task_classes)):
        classes = task_classes[task_id]
        within = F.log_softmax(scores[:, classes], dim=1)
        output[:, classes] = (
            within
            + float(task_weight) * task_log_probability[:, position, None]
        )
    return output


def _loaded_tensors(loaded, device):
    calibration = [value.to(device) for value in loaded["calibration"]]
    evaluation = [value.to(device) for value in loaded["evaluation"]]
    return calibration, evaluation


def evaluate_candidates(run_dir, output_dir, output_json, device="cuda:0"):
    loaded = _load_run(run_dir, output_dir, "validation", device)
    (_, cal_logits, cal_labels), (_, val_logits, val_labels) = (
        _loaded_tensors(loaded, device)
    )
    task_classes = loaded["task_classes"]
    records = []

    task_scores, task_fit = _task_affine_scores(
        cal_logits, cal_labels, val_logits, task_classes
    )
    task_summary = summarize_scores(task_scores, val_labels, task_classes)
    records.append({
        "candidate": "task_affine",
        "metrics": replace_final_metrics(loaded["results"], task_summary),
        "final_summary": task_summary,
        "fit": task_fit,
    })

    for class_reg in CLASS_REGULARIZATION:
        for task_reg in TASK_REGULARIZATION:
            state = fit_regularized_task_class_bias(
                cal_logits, cal_labels, task_classes, class_reg, task_reg
            )
            base_scores = apply_task_class_bias(val_logits, state)
            for task_weight in TASK_WEIGHTS:
                scores = hierarchical_task_scores(
                    base_scores, task_classes, task_weight
                )
                summary = summarize_scores(scores, val_labels, task_classes)
                records.append({
                    "candidate": candidate_name(
                        class_reg, task_reg, task_weight
                    ),
                    "metrics": replace_final_metrics(
                        loaded["results"], summary
                    ),
                    "final_summary": summary,
                    "fit": {
                        "class_regularization": class_reg,
                        "task_regularization": task_reg,
                        "task_weight": task_weight,
                        **state["fit_summary"],
                    },
                })

    output = {
        "run_dir": str(Path(run_dir).resolve()),
        "seed": int(loaded["config"]["seed"]),
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "evaluation_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "candidate_count": len(records),
        "diagnostic": {
            "task_affine": task_summary,
            "task_il_minus_class_il": (
                statistics.fmean(task_summary["task_il"].values())
                - task_summary["overall_accuracy"]
            ),
        },
        "calibration_audit": loaded["calibration_audit"],
        "selection_audit": loaded["selection_audit"],
        "records": records,
    }
    _write_json(output_json, output)
    return output


def _fit_selected(candidate, loaded, device):
    (_, cal_logits, cal_labels), (_, eval_logits, _) = (
        _loaded_tensors(loaded, device)
    )
    task_classes = loaded["task_classes"]
    if candidate == "task_affine":
        return _task_affine_scores(
            cal_logits, cal_labels, eval_logits, task_classes
        )[0]
    class_reg, task_reg, task_weight = parse_candidate(candidate)
    state = fit_regularized_task_class_bias(
        cal_logits, cal_labels, task_classes, class_reg, task_reg
    )
    scores = apply_task_class_bias(eval_logits, state)
    return hierarchical_task_scores(scores, task_classes, task_weight)


def evaluate_formal(run_dir, output_dir, selection_path, output_json,
                    device="cuda:0"):
    with open(selection_path, encoding="utf-8") as handle:
        selection = json.load(handle)
    if (
        not selection.get("passed")
        or selection.get("test_used_for_selection")
        or selection.get("bwt_floor") != -0.15
    ):
        raise ValueError("formal Stage C requires a legal frozen selection")
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
        "calibration_audit": loaded["calibration_audit"],
    }
    _write_json(output_json, output)
    return output


def summarize_formal(record_paths, selection_path, output_json):
    records = []
    for path in record_paths:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    if {int(item["seed"]) for item in records} != {42, 43, 44}:
        raise ValueError("formal summary requires seeds 42, 43, and 44")
    selection_sha = _sha256(selection_path)
    with open(selection_path, encoding="utf-8") as handle:
        candidate = json.load(handle)["selected_candidate"]
    if any(
        item["candidate"] != candidate
        or item["selection_sha256"] != selection_sha
        or item["selection_test_used"]
        or item["test_used_for_fit"]
        for item in records
    ):
        raise ValueError("formal records do not share the frozen selection")
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
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validation")
    validation.add_argument("--run-dir", required=True)
    validation.add_argument("--output-dir", required=True)
    validation.add_argument("--output-json", required=True)
    validation.add_argument("--device", default="cuda:0")
    select = commands.add_parser("select")
    select.add_argument("--records", nargs=3, required=True)
    select.add_argument("--output-json", required=True)
    select.add_argument("--bwt-floor", type=float, default=-0.15)
    formal = commands.add_parser("formal")
    formal.add_argument("--run-dir", required=True)
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--selection", required=True)
    formal.add_argument("--output-json", required=True)
    formal.add_argument("--device", default="cuda:0")
    aggregate = commands.add_parser("summarize")
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
