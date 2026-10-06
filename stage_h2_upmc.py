"""UPMC Stage H2: a frozen six-candidate TCB neighborhood."""
import argparse
import json
from pathlib import Path

from stage_h_calibration import (
    baseline_targets,
    cross_fit,
    load_validation_run,
    metrics_with_replaced_final,
    read_json,
    rounded_summary,
    select_candidate,
    sha256,
    stratified_fold_ids,
    write_json,
)


DATASET = "upmc_food101"
CANDIDATES = (
    {
        "name": "tcb_c0.001_t0.001_g1",
        "kind": "tcb",
        "class_reg": 0.001,
        "task_reg": 0.001,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.0003_t0.001_g1",
        "kind": "tcb",
        "class_reg": 0.0003,
        "task_reg": 0.001,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.003_t0.001_g1",
        "kind": "tcb",
        "class_reg": 0.003,
        "task_reg": 0.001,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.001_t0.0003_g1",
        "kind": "tcb",
        "class_reg": 0.001,
        "task_reg": 0.0003,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.001_t0.003_g1",
        "kind": "tcb",
        "class_reg": 0.001,
        "task_reg": 0.003,
        "task_weight": 1.0,
    },
    {
        "name": "tcb_c0.001_t0.001_g0.9",
        "kind": "tcb",
        "class_reg": 0.001,
        "task_reg": 0.001,
        "task_weight": 0.9,
    },
)


def find_record(report, candidate):
    matches = [
        item for item in report["records"] if item["candidate"] == candidate
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one prior record for {candidate}")
    return matches[0]


def evaluate(run_dir, baseline_summary_path, stage_h_report_path,
             output_dir, device):
    loaded = load_validation_run(run_dir, device)
    logits = loaded["logits"].to(device)
    labels = loaded["labels"].to(device)
    folds = stratified_fold_ids(labels.cpu().numpy())
    targets = baseline_targets(
        read_json(baseline_summary_path), DATASET,
        loaded["manifest"]["sha256"],
    )
    records = []
    for candidate in CANDIDATES:
        scores, fits = cross_fit(
            candidate, logits, labels, loaded["task_classes"], folds
        )
        final_summary = rounded_summary(
            scores, labels, loaded["task_classes"]
        )
        records.append({
            "candidate": candidate["name"],
            "spec": candidate,
            "metrics": metrics_with_replaced_final(
                loaded["results"], final_summary
            ),
            "final_summary": final_summary,
            "fold_fits": fits,
        })
    prior = read_json(stage_h_report_path)
    if prior["dataset"] != DATASET or prior["test_used_for_selection"]:
        raise ValueError("illegal Stage H source report")
    incumbent = CANDIDATES[0]["name"]
    incumbent_exact = (
        find_record(prior, incumbent)["metrics"]
        == find_record({"records": records}, incumbent)["metrics"]
    )
    raw_task_il = loaded["results"]["cl_metrics"]["AA_final_taskil"]
    selection = select_candidate(records, targets, raw_task_il)
    output_dir = Path(output_dir)
    report = {
        "passed": bool(incumbent_exact),
        "stage_h2_gate_passed": selection["passed"],
        "dataset": DATASET,
        "seed": int(loaded["config"]["seed"]),
        "candidate_count": len(records),
        "evaluation_source": "vector-train-validation-two-fold-crossfit",
        "test_used_for_selection": False,
        "checkpoint_sha256": sha256(loaded["checkpoint_path"]),
        "validation_manifest_sha256": loaded["manifest"]["sha256"],
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "stage_h_report_sha256": sha256(stage_h_report_path),
        "targets": targets,
        "selection": selection,
        "audit": {
            "incumbent_metrics_reproduced_exactly": incumbent_exact,
            "selection_audit": loaded["selection_audit"],
            "fold_counts": {
                str(fold): int((folds == fold).sum()) for fold in (0, 1)
            },
        },
        "records": records,
    }
    write_json(output_dir / "UPMC_STAGE_H2.json", report)
    if not report["passed"]:
        raise RuntimeError("UPMC Stage H2 audit failed")
    (output_dir / "UPMC_STAGE_H2_EXECUTION_SUCCESS").touch()
    if report["stage_h2_gate_passed"]:
        (output_dir / "UPMC_STAGE_H2_GATE_SUCCESS").touch()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--stage-h-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    report = evaluate(
        args.run_dir,
        args.baseline_summary,
        args.stage_h_report,
        args.output_dir,
        args.device,
    )
    print(json.dumps({
        "passed": report["passed"],
        "stage_h2_gate_passed": report["stage_h2_gate_passed"],
        "selection": report["selection"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
