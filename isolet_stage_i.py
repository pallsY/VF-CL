"""ISOLET Stage I: four frozen Ours training configurations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from dataset_protocol_smoke import replace_arg
from fair_main_table_3datasets import DATASETS, find_resume
from selected_protocol_seed42 import (
    audit_run,
    deterministic_env,
    discover_runs,
    find_complete,
    protocol_evidence,
    validation_command,
)
from stage_h_calibration import (
    baseline_targets,
    cross_fit,
    load_validation_run,
    metrics_with_replaced_final,
    read_json,
    rounded_summary,
    sha256,
    stratified_fold_ids,
    write_json,
)


DATASET = "isolet"
SPEC = "isolet:ours:42"
FIXED_TCB = {
    "name": "tcb_c0.001_t0.001_g1",
    "kind": "tcb",
    "class_reg": 0.001,
    "task_reg": 0.001,
    "task_weight": 1.0,
}
ISOLET_TASK_IL_FLOOR = 0.9813
CONFIGS = {
    "s0_incumbent": {},
    "s1_replay_ratio_1.5": {"--proto_replay_ratio": "1.5"},
    "s2_proto_lambda_0.25": {"--proto_lambda_a": "0.25"},
    "s3_distill_0.35": {"--distill_weight": "0.35"},
}


def parse_candidate(candidate):
    if candidate not in CONFIGS or candidate == "s0_incumbent":
        raise ValueError(f"unknown runnable Stage I candidate: {candidate}")
    return candidate


def stage_i_command(candidate, device, results_dir, resume_run_dir=None):
    parse_candidate(candidate)
    command = validation_command(
        SPEC, device, results_dir, resume_run_dir=resume_run_dir
    )
    replace_arg(command, "--exp_name", f"isolet_stage_i_{candidate}")
    for option, value in CONFIGS[candidate].items():
        replace_arg(command, option, value)
    return command


def calibrated_record(candidate, training_record, run_dir, baseline_summary,
                      device):
    loaded = load_validation_run(run_dir, device)
    logits = loaded["logits"].to(device)
    labels = loaded["labels"].to(device)
    folds = stratified_fold_ids(labels.cpu().numpy())
    scores, fits = cross_fit(
        FIXED_TCB, logits, labels, loaded["task_classes"], folds
    )
    final_summary = rounded_summary(scores, labels, loaded["task_classes"])
    metrics = metrics_with_replaced_final(loaded["results"], final_summary)
    targets = baseline_targets(
        read_json(baseline_summary), DATASET, loaded["manifest"]["sha256"]
    )
    task_il_floor = ISOLET_TASK_IL_FLOOR
    gates = {
        "bwt": metrics["BWT"] >= targets["er_bwt"],
        "aa_final": metrics["AA_final"] > targets["der_pp_aa_final"],
        "task_il": metrics["AA_final_taskil"] >= task_il_floor,
    }
    return {
        "candidate": candidate,
        "training_overrides": CONFIGS[candidate],
        "training_record": training_record,
        "run_dir": str(run_dir),
        "fixed_calibration": FIXED_TCB,
        "metrics": metrics,
        "final_summary": final_summary,
        "fold_fits": fits,
        "targets": {
            **targets,
            "task_il_floor": task_il_floor,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "test_used_for_selection": False,
        "validation_manifest_sha256": loaded["manifest"]["sha256"],
        "checkpoint_sha256": sha256(loaded["checkpoint_path"]),
    }


def run_job(candidate, device, matrix_root, baseline_summary):
    parse_candidate(candidate)
    root = Path(matrix_root) / "runs" / candidate
    root.mkdir(parents=True, exist_ok=True)
    if (root / "SUCCESS").is_file() and (root / "record.json").is_file():
        print(f"SKIP complete {candidate}")
        return 0
    expected_tasks = len(DATASETS[DATASET]["tasks"])
    complete = find_complete(root, expected_tasks)
    resume = None if complete else find_resume(discover_runs(root))
    command = stage_i_command(
        candidate, device, root, resume_run_dir=resume
    )
    write_json(root / "planned_protocol.json", {
        "candidate": candidate,
        "selection_source": "training-validation",
        "selection_test_used": False,
        "conditional_stage": "S3 runs only if S1 and S2 both fail",
        "fixed_calibration": FIXED_TCB,
        "frozen_ce_evidence": protocol_evidence(DATASET),
        "command": command,
    })
    if not complete:
        with open(root / "job.log", "a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=deterministic_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            print(
                f"FAILED {candidate}: exit={completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
        complete = find_complete(root, expected_tasks)
    if not complete:
        print(f"FAILED {candidate}: no complete run", file=sys.stderr)
        return 90
    run_dir = complete[-1]
    try:
        training_record = audit_run(SPEC, command, run_dir)
        record = calibrated_record(
            candidate, training_record, run_dir, baseline_summary, device
        )
    except Exception as error:
        print(f"FAILED {candidate}: {error}", file=sys.stderr)
        return 91
    write_json(root / "record.json", record)
    (root / "SUCCESS").touch()
    print(
        f"COMPLETE {candidate}: AA={record['metrics']['AA_final']:.4f} "
        f"BWT={record['metrics']['BWT']:.4f} "
        f"TIL={record['metrics']['AA_final_taskil']:.4f} "
        f"PASS={record['passed']}"
    )
    return 0


def incumbent_record(stage_h_report):
    report = read_json(stage_h_report)
    matches = [
        item for item in report["records"]
        if item["candidate"] == FIXED_TCB["name"]
    ]
    if report["dataset"] != DATASET or len(matches) != 1:
        raise ValueError("invalid ISOLET Stage H incumbent report")
    metrics = matches[0]["metrics"]
    targets = report["targets"]
    task_il_floor = float(report["selection"]["task_il_floor"])
    gates = {
        "bwt": metrics["BWT"] >= targets["er_bwt"],
        "aa_final": metrics["AA_final"] > targets["der_pp_aa_final"],
        "task_il": metrics["AA_final_taskil"] >= task_il_floor,
    }
    return {
        "candidate": "s0_incumbent",
        "training_overrides": {},
        "metrics": metrics,
        "targets": {**targets, "task_il_floor": task_il_floor},
        "gates": gates,
        "passed": all(gates.values()),
        "source_stage_h_sha256": sha256(stage_h_report),
        "test_used_for_selection": False,
    }


def should_run_s3(completed, records):
    first_pair_complete = all(
        name in completed
        for name in ("s1_replay_ratio_1.5", "s2_proto_lambda_0.25")
    )
    return (
        first_pair_complete
        and "s3_distill_0.35" not in completed
        and not any(item["passed"] for item in records)
    )


def summarize(matrix_root, stage_h_report):
    matrix_root = Path(matrix_root)
    records = [incumbent_record(stage_h_report)]
    completed = []
    for candidate in CONFIGS:
        if candidate == "s0_incumbent":
            continue
        path = matrix_root / "runs" / candidate / "record.json"
        if path.is_file():
            records.append(read_json(path))
            completed.append(candidate)
    eligible = [
        item for item in records
        if item["gates"]["bwt"] and item["gates"]["task_il"]
    ]
    selected = max(
        eligible,
        key=lambda item: (item["metrics"]["AA_final"], item["metrics"]["BWT"]),
        default=None,
    )
    gate_passed = any(item["passed"] for item in records)
    run_s3 = should_run_s3(completed, records)
    output = {
        "passed": True,
        "stage_i_gate_passed": gate_passed,
        "test_used_for_selection": False,
        "completed_candidates": completed,
        "run_s3": run_s3,
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_metrics": None if selected is None else selected["metrics"],
        "records": records,
    }
    write_json(matrix_root / "ISOLET_STAGE_I_SUMMARY.json", output)
    (matrix_root / "ISOLET_STAGE_I_EXECUTION_SUCCESS").touch()
    if gate_passed:
        (matrix_root / "ISOLET_STAGE_I_GATE_SUCCESS").touch()
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-job")
    run.add_argument("candidate")
    run.add_argument("--device", required=True)
    run.add_argument("--matrix-root", required=True)
    run.add_argument("--baseline-summary", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--matrix-root", required=True)
    summary.add_argument("--stage-h-report", required=True)
    args = parser.parse_args()
    if args.command == "run-job":
        raise SystemExit(run_job(
            args.candidate,
            args.device,
            args.matrix_root,
            args.baseline_summary,
        ))
    output = summarize(args.matrix_root, args.stage_h_report)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
