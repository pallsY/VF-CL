"""Stage H: cross-fitted Class-IL calibration on frozen validation runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_output_bias import (
    _load_checkpoint,
    configure_evaluation_determinism,
)
from bic_calibration import fit_final_calibrator
from cifar100_stage_c_head import (
    fit_regularized_task_class_bias,
    hierarchical_task_scores,
)
from data_utils import VFLDataset
from metrics import MetricsTracker
from models import build_models
from offline_balanced_head import (
    apply_task_class_bias,
    collect_embeddings_logits,
    summarize_scores,
)
from vfl_trainer import VFLTrainer


FOLD_SEED = 20260809
TASK_IL_TOLERANCE = 0.01
CANDIDATES = (
    {"name": "identity", "kind": "identity"},
    {"name": "prior_tau_0.5", "kind": "prior", "tau": 0.5},
    {"name": "prior_tau_1", "kind": "prior", "tau": 1.0},
    {"name": "affine_beta", "kind": "affine", "mode": "beta_only"},
    {
        "name": "affine_joint",
        "kind": "affine",
        "mode": "joint_alpha_beta",
    },
    {
        "name": "tcb_c0.001_t0.001_g1",
        "kind": "tcb",
        "class_reg": 0.001,
        "task_reg": 0.001,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.01_t0.01_g1.15",
        "kind": "tcb",
        "class_reg": 0.01,
        "task_reg": 0.01,
        "task_weight": 1.15,
    },
    {
        "name": "tcb_c0.01_t0.01_g1.3",
        "kind": "tcb",
        "class_reg": 0.01,
        "task_reg": 0.01,
        "task_weight": 1.3,
    },
)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_checkpoint_for_config(run_dir, config):
    task_id = int(config["num_tasks"]) - 1
    if task_id < 0:
        raise ValueError("num_tasks must be positive")
    path = Path(run_dir) / "checkpoints" / f"event_{task_id}_CIL.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"full final CIL checkpoint is required: {path}"
        )
    return path


def task_classes_from_config(config):
    custom = str(config.get("custom_tasks") or "").strip()
    if custom:
        tasks = {
            task_id: [int(value) for value in group.split(",") if value != ""]
            for task_id, group in enumerate(custom.split("|"))
        }
    else:
        per_task = int(config["classes_per_task"])
        num_classes = int(config["num_classes"])
        tasks = {
            task_id: list(range(
                task_id * per_task,
                min((task_id + 1) * per_task, num_classes),
            ))
            for task_id in range(int(config["num_tasks"]))
        }
    flattened = [value for task_id in sorted(tasks) for value in tasks[task_id]]
    expected = list(range(int(config["num_classes"])))
    if flattened != expected or any(not values for values in tasks.values()):
        raise ValueError("task classes must cover contiguous classes exactly once")
    if len(tasks) != int(config["num_tasks"]):
        raise ValueError("task count does not match config")
    return tasks


def stratified_fold_ids(labels, seed=FOLD_SEED):
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.full(labels.shape, -1, dtype=np.int8)
    for class_id in sorted(np.unique(labels).tolist()):
        rows = np.flatnonzero(labels == class_id)
        if len(rows) < 2:
            raise ValueError("each class needs at least two validation rows")
        shuffled = np.random.default_rng(int(seed) + int(class_id)).permutation(rows)
        folds[shuffled[::2]] = 0
        folds[shuffled[1::2]] = 1
    if np.any(folds < 0) or set(folds.tolist()) != {0, 1}:
        raise ValueError("invalid cross-fit fold assignment")
    return folds


def rounded_summary(scores, labels, task_classes):
    summary = summarize_scores(scores, labels, task_classes)
    return {
        "overall_accuracy": round(float(summary["overall_accuracy"]), 6),
        "per_task_accuracy": {
            key: round(float(value), 4)
            for key, value in summary["per_task_accuracy"].items()
        },
        "task_il": {
            key: round(float(value), 4)
            for key, value in summary["task_il"].items()
        },
        "task_prediction_fraction": {
            key: round(float(value), 6)
            for key, value in summary["task_prediction_fraction"].items()
        },
    }


def metrics_with_replaced_final(results, final_summary):
    history = copy.deepcopy(results["task_acc_history"])
    if not history or "CIL" not in history[-1]["step"]:
        raise ValueError("results must end with a CIL event")
    history[-1]["per_task_accs"] = final_summary["per_task_accuracy"]
    history[-1]["per_task_accs_taskil"] = final_summary["task_il"]
    history[-1]["overall_acc"] = final_summary["overall_accuracy"]
    tracker = MetricsTracker()
    tracker.load_dict({"task_acc_history": history})
    metrics = tracker.compute_cl_metrics()
    return {
        key: float(metrics[key])
        for key in ("AA_final", "AA_cil", "BWT", "AA_final_taskil")
    }


def fit_apply(candidate, train_logits, train_labels, evaluation_logits,
              task_classes):
    kind = candidate["kind"]
    if kind == "identity":
        return evaluation_logits.clone(), {"fit_source": "none"}
    if kind == "prior":
        tau = float(candidate["tau"])
        prior = torch.softmax(train_logits, dim=1).mean(dim=0).clamp(min=1e-12)
        return evaluation_logits - tau * prior.log(), {
            "tau": tau,
            "prior_min": float(prior.min()),
            "prior_max": float(prior.max()),
        }
    if kind == "affine":
        calibrator, fit = fit_final_calibrator(
            train_logits,
            train_labels,
            task_classes,
            candidate["mode"],
            lr=0.05,
            steps=1000,
        )
        return calibrator.apply(evaluation_logits), fit
    if kind == "tcb":
        state = fit_regularized_task_class_bias(
            train_logits,
            train_labels,
            task_classes,
            candidate["class_reg"],
            candidate["task_reg"],
            steps=600,
            lr=0.03,
        )
        scores = apply_task_class_bias(evaluation_logits, state)
        scores = hierarchical_task_scores(
            scores, task_classes, candidate["task_weight"]
        )
        return scores, {
            "class_regularization": candidate["class_reg"],
            "task_regularization": candidate["task_reg"],
            "task_weight": candidate["task_weight"],
            **state["fit_summary"],
        }
    raise ValueError(f"unknown candidate kind: {kind}")


def cross_fit(candidate, logits, labels, task_classes, folds):
    scores = torch.empty_like(logits)
    fits = []
    for evaluation_fold in (0, 1):
        evaluation_rows = torch.as_tensor(folds == evaluation_fold)
        training_rows = ~evaluation_rows
        fold_scores, fit = fit_apply(
            candidate,
            logits[training_rows],
            labels[training_rows],
            logits[evaluation_rows],
            task_classes,
        )
        scores[evaluation_rows] = fold_scores
        fits.append({
            "evaluation_fold": evaluation_fold,
            "fit_count": int(training_rows.sum()),
            "evaluation_count": int(evaluation_rows.sum()),
            "fit": fit,
        })
    if not torch.isfinite(scores).all():
        raise ValueError("cross-fitted scores contain non-finite values")
    return scores, fits


def baseline_targets(summary, dataset, manifest_sha256):
    if not summary.get("passed") or summary.get("selection_test_used"):
        raise ValueError("baseline summary did not pass its selection audit")
    manifest = summary["manifest_checks"][dataset]
    if not manifest.get("passed") or manifest.get("unique_hashes") != [manifest_sha256]:
        raise ValueError("baseline and Stage H validation manifests differ")
    by_method = {
        item["method"]: item for item in summary["rankings"][dataset]
    }
    return {
        "er_bwt": float(by_method["er"]["BWT"]),
        "der_pp_aa_final": float(by_method["der_pp"]["AA_final"]),
    }


def select_candidate(records, targets, raw_task_il,
                     task_il_tolerance=TASK_IL_TOLERANCE):
    eligible = [
        record for record in records
        if record["metrics"]["BWT"] >= targets["er_bwt"]
        and record["metrics"]["AA_final_taskil"] >= (
            float(raw_task_il) - float(task_il_tolerance)
        )
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
        "task_il_floor": float(raw_task_il) - float(task_il_tolerance),
        "aa_final_target": targets["der_pp_aa_final"],
        "passed": (
            selected is not None
            and selected["metrics"]["AA_final"] > targets["der_pp_aa_final"]
        ),
    }


def load_validation_run(run_dir, device):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    results = read_json(run_dir / "results.json")
    configure_evaluation_determinism(int(config["seed"]))
    values = dict(config)
    values.update(device=device, num_workers=0, output_dir=str(run_dir))
    args = argparse.Namespace(**values)
    dataset = VFLDataset(args)
    manifest_path = run_dir / "validation" / "validation_manifest.json"
    manifest = read_json(manifest_path)
    if dataset.validation_manifest != manifest:
        raise ValueError("validation manifest replay mismatch")
    audit = dataset.selection_audit()
    if not audit.get("passed") or audit.get("test_used_for_selection"):
        raise ValueError("validation selection audit failed")
    if audit.get("evaluation_source") != "vector-train-validation":
        raise ValueError("Stage H requires vector training-validation")

    checkpoint_path = final_checkpoint_for_config(run_dir, config)
    checkpoint = _load_checkpoint(checkpoint_path)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(checkpoint["trainer_state"])
    classes = list(range(int(config["num_classes"])))
    loader = dataset.get_validation_loader(classes)
    embeddings, logits, labels = collect_embeddings_logits(trainer, loader, args)
    saved = np.load(run_dir / "final_probs.npz")
    saved_labels = saved["labels"]
    saved_probs = saved["probs"]
    replay_probs = torch.softmax(logits, dim=1).numpy()
    if not np.array_equal(saved_labels, labels.numpy()):
        raise ValueError("validation label replay mismatch")
    if not np.array_equal(saved_probs.argmax(1), replay_probs.argmax(1)):
        raise ValueError("validation prediction replay mismatch")
    if not np.allclose(saved_probs, replay_probs, rtol=1e-5, atol=1e-6):
        raise ValueError("validation probability replay mismatch")
    return {
        "config": config,
        "results": results,
        "task_classes": task_classes_from_config(config),
        "embeddings": embeddings.float(),
        "logits": logits.float(),
        "labels": labels.long(),
        "initial_weight": trainer.top_model.classifier.weight.detach().cpu(),
        "initial_bias": trainer.top_model.classifier.bias.detach().cpu(),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "checkpoint_path": checkpoint_path,
        "selection_audit": audit,
    }


def evaluate(dataset, run_dir, baseline_summary_path, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_validation_run(run_dir, device)
    logits = loaded["logits"].to(device)
    labels = loaded["labels"].to(device)
    folds = stratified_fold_ids(labels.cpu().numpy())
    targets = baseline_targets(
        read_json(baseline_summary_path), dataset, loaded["manifest"]["sha256"]
    )
    records = []
    for candidate in CANDIDATES:
        scores, fits = cross_fit(
            candidate, logits, labels, loaded["task_classes"], folds
        )
        final_summary = rounded_summary(scores, labels, loaded["task_classes"])
        records.append({
            "candidate": candidate["name"],
            "spec": candidate,
            "metrics": metrics_with_replaced_final(
                loaded["results"], final_summary
            ),
            "final_summary": final_summary,
            "fold_fits": fits,
        })
    raw = loaded["results"]["cl_metrics"]
    identity = records[0]["metrics"]
    replay_checks = {
        "aa_final_exact": identity["AA_final"] == float(raw["AA_final"]),
        "bwt_exact": identity["BWT"] == float(raw["BWT"]),
        "task_il_exact": (
            identity["AA_final_taskil"] == float(raw["AA_final_taskil"])
        ),
    }
    selection = select_candidate(
        records, targets, raw["AA_final_taskil"]
    )
    report = {
        "passed": all(replay_checks.values()),
        "stage_h_gate_passed": selection["passed"],
        "dataset": dataset,
        "seed": int(loaded["config"]["seed"]),
        "candidate_count": len(records),
        "evaluation_source": "vector-train-validation-two-fold-crossfit",
        "test_used_for_selection": False,
        "fold_seed": FOLD_SEED,
        "checkpoint_path": str(loaded["checkpoint_path"]),
        "checkpoint_sha256": sha256(loaded["checkpoint_path"]),
        "validation_manifest_path": str(loaded["manifest_path"]),
        "validation_manifest_sha256": loaded["manifest"]["sha256"],
        "baseline_summary_path": str(Path(baseline_summary_path)),
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "targets": targets,
        "selection": selection,
        "audit": {
            "selection_audit": loaded["selection_audit"],
            "replay_checks": replay_checks,
            "fold_counts": {
                str(fold): int(np.sum(folds == fold)) for fold in (0, 1)
            },
        },
        "records": records,
    }
    output_path = output_dir / f"{dataset}_stage_h.json"
    write_json(output_path, report)
    if not report["passed"]:
        raise RuntimeError(f"{dataset} Stage H replay audit failed")
    (output_dir / f"{dataset}_STAGE_H_SUCCESS").touch()
    return report


def summarize_reports(record_paths, output_dir):
    reports = [read_json(path) for path in record_paths]
    if {item["dataset"] for item in reports} != {"isolet", "upmc_food101"}:
        raise ValueError("Stage H summary requires ISOLET and UPMC reports")
    output = {
        "passed": all(item["passed"] for item in reports),
        "stage_h_gate_passed": all(
            item["stage_h_gate_passed"] for item in reports
        ),
        "test_used_for_selection": any(
            item["test_used_for_selection"] for item in reports
        ),
        "datasets": {
            item["dataset"]: {
                "selection": item["selection"],
                "targets": item["targets"],
            }
            for item in reports
        },
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "STAGE_H_SUMMARY.json", output)
    if output["passed"] and not output["test_used_for_selection"]:
        (output_dir / "STAGE_H_EXECUTION_SUCCESS").touch()
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--dataset", choices=("isolet", "upmc_food101"), required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--baseline-summary", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--records", nargs=2, required=True)
    summary.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate(
            args.dataset,
            args.run_dir,
            args.baseline_summary,
            args.output_dir,
            args.device,
        )
        printable = {
            "passed": result["passed"],
            "stage_h_gate_passed": result["stage_h_gate_passed"],
            "dataset": result["dataset"],
            "selection": result["selection"],
        }
    else:
        printable = summarize_reports(args.records, args.output_dir)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
