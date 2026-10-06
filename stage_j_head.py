"""Stage J: cross-fitted representation-aligned final heads."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from offline_balanced_head import (
    apply_linear_head,
    cosine_scores,
    fit_cosine_centroids,
    fit_linear_prior,
)
from stage_h_calibration import (
    baseline_targets,
    fit_apply,
    load_validation_run,
    metrics_with_replaced_final,
    read_json,
    rounded_summary,
    sha256,
    stratified_fold_ids,
    write_json,
)


CANDIDATES = (
    "j0_tcb",
    "j1_cosine_ncm",
    "j2_l2_linear",
    "j3_tcb_ncm_blend",
)
PROTOCOLS = {
    "isolet": {
        "incumbent_name": "tcb_c0.001_t0.001_g1",
        "incumbent": {
            "name": "tcb_c0.001_t0.001_g1",
            "kind": "tcb",
            "class_reg": 0.001,
            "task_reg": 0.001,
            "task_weight": 1.0,
        },
        "task_il_floor": 0.9813,
    },
    "upmc_food101": {
        "incumbent_name": "tcb_c0.001_t0.003_g1",
        "incumbent": {
            "name": "tcb_c0.001_t0.003_g1",
            "kind": "tcb",
            "class_reg": 0.001,
            "task_reg": 0.003,
            "task_weight": 1.0,
        },
        "task_il_floor": 0.9195,
    },
}
LINEAR_REGULARIZATION = 0.01
BLEND_WEIGHT = 0.5


def ordered_classes(task_classes):
    return [
        class_id
        for task_id in sorted(task_classes)
        for class_id in task_classes[task_id]
    ]


def row_standardize(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True
    ).clamp(min=1e-6)


def ncm_scores(train_embeddings, train_labels, evaluation_embeddings,
               evaluation_logits, task_classes):
    classes = ordered_classes(task_classes)
    centroids = fit_cosine_centroids(
        train_embeddings, train_labels, classes
    )
    scores = evaluation_logits.clone()
    scores[:, classes] = cosine_scores(evaluation_embeddings, centroids)
    return scores, {
        "centroid_count": len(classes),
        "embedding_dim": int(train_embeddings.shape[1]),
    }


def fit_apply_head(candidate, dataset, train_embeddings, train_logits,
                   train_labels, evaluation_embeddings, evaluation_logits,
                   task_classes, initial_weight, initial_bias):
    protocol = PROTOCOLS[dataset]
    if candidate == "j0_tcb":
        return fit_apply(
            protocol["incumbent"],
            train_logits,
            train_labels,
            evaluation_logits,
            task_classes,
        )
    if candidate == "j1_cosine_ncm":
        return ncm_scores(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            evaluation_logits,
            task_classes,
        )
    if candidate == "j2_l2_linear":
        state = fit_linear_prior(
            train_embeddings,
            train_labels,
            initial_weight,
            initial_bias,
            regularization=LINEAR_REGULARIZATION,
            steps=500,
            lr=0.01,
        )
        return apply_linear_head(evaluation_embeddings, state), {
            "regularization": LINEAR_REGULARIZATION,
            "weight_delta_norm": float(
                (state["weight"] - initial_weight.cpu()).norm()
            ),
            "bias_delta_norm": float(
                (state["bias"] - initial_bias.cpu()).norm()
            ),
        }
    if candidate == "j3_tcb_ncm_blend":
        tcb, tcb_fit = fit_apply(
            protocol["incumbent"],
            train_logits,
            train_labels,
            evaluation_logits,
            task_classes,
        )
        ncm, ncm_fit = ncm_scores(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            evaluation_logits,
            task_classes,
        )
        classes = ordered_classes(task_classes)
        scores = evaluation_logits.clone()
        scores[:, classes] = (
            BLEND_WEIGHT * row_standardize(tcb[:, classes])
            + (1.0 - BLEND_WEIGHT) * row_standardize(ncm[:, classes])
        )
        return scores, {
            "tcb": tcb_fit,
            "ncm": ncm_fit,
            "tcb_weight": BLEND_WEIGHT,
            "ncm_weight": 1.0 - BLEND_WEIGHT,
        }
    raise ValueError(f"unknown Stage J candidate: {candidate}")


def cross_fit_head(candidate, dataset, loaded, folds, device):
    embeddings = loaded["embeddings"].to(device)
    logits = loaded["logits"].to(device)
    labels = loaded["labels"].to(device)
    scores = torch.empty_like(logits)
    fits = []
    for evaluation_fold in (0, 1):
        evaluation_rows = torch.as_tensor(folds == evaluation_fold)
        training_rows = ~evaluation_rows
        fold_scores, fit = fit_apply_head(
            candidate,
            dataset,
            embeddings[training_rows],
            logits[training_rows],
            labels[training_rows],
            embeddings[evaluation_rows],
            logits[evaluation_rows],
            loaded["task_classes"],
            loaded["initial_weight"],
            loaded["initial_bias"],
        )
        scores[evaluation_rows] = fold_scores
        fits.append({
            "evaluation_fold": evaluation_fold,
            "fit_count": int(training_rows.sum()),
            "evaluation_count": int(evaluation_rows.sum()),
            "fit": fit,
        })
    if not torch.isfinite(scores).all():
        raise ValueError("Stage J produced non-finite scores")
    return scores, fits


def prior_incumbent(report, dataset):
    protocol = PROTOCOLS[dataset]
    if report.get("dataset") != dataset or report.get("test_used_for_selection"):
        raise ValueError("illegal incumbent report")
    matches = [
        item for item in report["records"]
        if item["candidate"] == protocol["incumbent_name"]
    ]
    if len(matches) != 1:
        raise ValueError("incumbent report does not contain the frozen TCB")
    return matches[0]


def select(records, targets, task_il_floor):
    eligible = [
        item for item in records
        if item["metrics"]["BWT"] >= targets["er_bwt"]
        and item["metrics"]["AA_final_taskil"] >= task_il_floor
    ]
    selected = max(
        eligible,
        key=lambda item: (item["metrics"]["AA_final"], item["metrics"]["BWT"]),
        default=None,
    )
    return {
        "eligible_candidates": [item["candidate"] for item in eligible],
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_metrics": None if selected is None else selected["metrics"],
        "bwt_gate": targets["er_bwt"],
        "aa_final_target": targets["der_pp_aa_final"],
        "task_il_floor": task_il_floor,
        "passed": (
            selected is not None
            and selected["metrics"]["AA_final"] > targets["der_pp_aa_final"]
        ),
    }


def evaluate(dataset, run_dir, baseline_summary_path, incumbent_report_path,
             output_dir, device):
    loaded = load_validation_run(run_dir, device)
    labels = loaded["labels"].to(device)
    folds = stratified_fold_ids(labels.cpu().numpy())
    targets = baseline_targets(
        read_json(baseline_summary_path), dataset,
        loaded["manifest"]["sha256"],
    )
    records = []
    for candidate in CANDIDATES:
        scores, fits = cross_fit_head(
            candidate, dataset, loaded, folds, device
        )
        final_summary = rounded_summary(
            scores, labels, loaded["task_classes"]
        )
        records.append({
            "candidate": candidate,
            "metrics": metrics_with_replaced_final(
                loaded["results"], final_summary
            ),
            "final_summary": final_summary,
            "fold_fits": fits,
        })
    prior = prior_incumbent(read_json(incumbent_report_path), dataset)
    incumbent_exact = records[0]["metrics"] == prior["metrics"]
    task_il_floor = PROTOCOLS[dataset]["task_il_floor"]
    selection = select(records, targets, task_il_floor)
    report = {
        "passed": bool(incumbent_exact),
        "stage_j_gate_passed": selection["passed"],
        "dataset": dataset,
        "seed": int(loaded["config"]["seed"]),
        "candidate_count": len(records),
        "evaluation_source": "vector-train-validation-two-fold-crossfit",
        "test_used_for_selection": False,
        "checkpoint_sha256": sha256(loaded["checkpoint_path"]),
        "validation_manifest_sha256": loaded["manifest"]["sha256"],
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "incumbent_report_sha256": sha256(incumbent_report_path),
        "targets": targets,
        "selection": selection,
        "audit": {
            "incumbent_metrics_reproduced_exactly": incumbent_exact,
            "selection_audit": loaded["selection_audit"],
            "fold_counts": {
                str(fold): int(np.sum(folds == fold)) for fold in (0, 1)
            },
        },
        "records": records,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / f"{dataset}_STAGE_J.json", report)
    if not report["passed"]:
        raise RuntimeError(f"{dataset} Stage J audit failed")
    (output_dir / f"{dataset}_STAGE_J_EXECUTION_SUCCESS").touch()
    if report["stage_j_gate_passed"]:
        (output_dir / f"{dataset}_STAGE_J_GATE_SUCCESS").touch()
    return report


def summarize(record_paths, output_dir):
    reports = [read_json(path) for path in record_paths]
    if {item["dataset"] for item in reports} != set(PROTOCOLS):
        raise ValueError("Stage J summary requires ISOLET and UPMC")
    output = {
        "passed": all(item["passed"] for item in reports),
        "stage_j_gate_passed": all(
            item["stage_j_gate_passed"] for item in reports
        ),
        "test_used_for_selection": any(
            item["test_used_for_selection"] for item in reports
        ),
        "datasets": {
            item["dataset"]: item["selection"] for item in reports
        },
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "STAGE_J_SUMMARY.json", output)
    if output["passed"] and not output["test_used_for_selection"]:
        (output_dir / "STAGE_J_EXECUTION_SUCCESS").touch()
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--dataset", choices=tuple(PROTOCOLS), required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--baseline-summary", required=True)
    run.add_argument("--incumbent-report", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--records", nargs=2, required=True)
    summary.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        report = evaluate(
            args.dataset,
            args.run_dir,
            args.baseline_summary,
            args.incumbent_report,
            args.output_dir,
            args.device,
        )
        output = {
            "dataset": report["dataset"],
            "passed": report["passed"],
            "stage_j_gate_passed": report["stage_j_gate_passed"],
            "selection": report["selection"],
        }
    else:
        output = summarize(args.records, args.output_dir)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
