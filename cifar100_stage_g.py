#!/usr/bin/env python3
"""Stage G late-task supervised-contrastive validation pipeline."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from cifar100_stage_f import (
    FROZEN_HEAD,
    METRICS,
    _normalized,
    _read_json,
    _write_json,
    claim_next,
    eligible,
    expected_config as stage_f_config,
)
from formal_cifar100_metrics import calibration_history, calibration_is_legal, compute_formal_metrics


CANDIDATES = {
    "supcon_002": 0.02,
    "supcon_005": 0.05,
    "supcon_010": 0.10,
}
SEEDS = (42, 43, 44)


def parse_job(spec):
    candidate, seed = spec.rsplit(":", 1)
    if candidate not in CANDIDATES or int(seed) not in SEEDS:
        raise ValueError(f"unknown Stage G job: {spec}")
    return candidate, int(seed)


def stage_a_jobs():
    return [f"{candidate}:42" for candidate in CANDIDATES]


def stage_b_jobs(stage_a_summary):
    summary = _read_json(stage_a_summary)
    if summary.get("test_used_for_selection") or summary.get("selection_source") != "cifar100-train-validation":
        raise ValueError("illegal Stage G A summary")
    return [
        f"{candidate}:{seed}"
        for candidate in summary.get("promoted_candidates", [])
        for seed in (43, 44)
    ]


def expected_config(spec):
    candidate, seed = parse_job(spec)
    config = stage_f_config(f"decay_005:{seed}")
    config.update({
        "current_supcon_weight": CANDIDATES[candidate],
        "current_supcon_temperature": 0.1,
        "current_supcon_start_task": 5,
    })
    return config


def command_for(spec, study_root, repo_root, python):
    candidate, seed = parse_job(spec)
    values = expected_config(spec)
    values["results_dir"] = str(Path(study_root) / "runs")
    values["exp_name"] = f"cifar100_stage_g_validation_{candidate}_seed{seed}"
    command = [str(python), str(Path(repo_root) / "main.py")]
    for key, value in values.items():
        if key == "unlearn_after_tasks":
            value = ",".join(map(str, value))
        elif key == "unlearn_classes":
            value = ";".join(",".join(map(str, group)) for group in value)
        command.extend((f"--{key}", str(value)))
    return command


def _prefix(spec):
    candidate, seed = parse_job(spec)
    return f"cifar100_stage_g_validation_{candidate}_seed{seed}_"


def find_run(spec, study_root, complete=True):
    paths = sorted(
        (Path(study_root) / "runs").glob(_prefix(spec) + "*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in paths:
        has_config = (run_dir / "config.json").is_file()
        has_result = (run_dir / "results.json").is_file()
        if (complete and has_result) or (not complete and has_config and not has_result):
            return run_dir
    raise FileNotFoundError(spec)


def audit_run(spec, run_dir, head_json, code_commit):
    run_dir = Path(run_dir)
    expected = expected_config(spec)
    actual = _normalized(_read_json(run_dir / "config.json"), expected)
    if actual != expected:
        mismatch = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
        raise ValueError(f"{spec}: protocol mismatch {mismatch}")
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    result_path = run_dir / "results.json"
    if not checkpoint.is_file() or not checkpoint.stat().st_size or not result_path.is_file():
        raise ValueError(f"{spec}: incomplete training artifacts")
    result = _read_json(result_path)
    selection = result.get("selection_audit", {})
    overlap_keys = (
        "training_calibration_overlap_count",
        "training_validation_overlap_count",
        "calibration_validation_overlap_count",
    )
    if (
        not selection.get("passed")
        or selection.get("test_used_for_selection")
        or selection.get("evaluation_source") != "cifar100-train-validation"
        or any(int(selection.get(key, -1)) != 0 for key in overlap_keys)
    ):
        raise ValueError(f"{spec}: illegal validation selection audit")
    events = result.get("bic_history") or []
    if len(events) != 10 or not calibration_is_legal(events):
        raise ValueError(f"{spec}: illegal calibrated trajectory")
    before_head = compute_formal_metrics(calibration_history(events), expected_tasks=10)

    head = _read_json(head_json)
    candidate, seed = parse_job(spec)
    if (
        int(head.get("seed", -1)) != seed
        or Path(head.get("run_dir", "")).resolve() != run_dir.resolve()
        or head.get("evaluation_source") != "cifar100-train-validation"
        or head.get("test_used_for_selection")
    ):
        raise ValueError(f"{spec}: illegal fixed-head evaluation")
    rows = [row for row in head.get("records", []) if row.get("candidate") == FROZEN_HEAD]
    if len(rows) != 1:
        raise ValueError(f"{spec}: frozen head candidate missing")
    metrics = {key: float(rows[0]["metrics"][key]) for key in METRICS}
    record = {
        "job": spec,
        "candidate": candidate,
        "seed": seed,
        "supcon_weight": CANDIDATES[candidate],
        "frozen_head": FROZEN_HEAD,
        "code_commit": code_commit,
        "config": expected,
        "run_dir": str(run_dir.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "head_json": str(Path(head_json).resolve()),
        "metrics_before_frozen_head": before_head,
        "metrics": metrics,
        "test_used_for_selection": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def select_stage_a(records, aa_floor=0.37, bwt_floor=-0.18, task_il_floor=0.755):
    passing = [row for row in records if eligible(row["metrics"], aa_floor, bwt_floor, task_il_floor)]
    passing.sort(key=lambda row: (
        row["metrics"]["aa_final_cil"], row["metrics"]["task_il_final"], row["metrics"]["bwt_cil"]
    ), reverse=True)
    return [row["candidate"] for row in passing[:2]]


def summarize_rows(records):
    rows = []
    for candidate in sorted({row["candidate"] for row in records}):
        candidate_records = [row for row in records if row["candidate"] == candidate]
        if {row["seed"] for row in candidate_records} != set(SEEDS):
            raise ValueError(f"{candidate}: Stage G B requires seeds 42, 43, and 44")
        summary = {"candidate": candidate, "supcon_weight": CANDIDATES[candidate]}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in candidate_records]
            summary[metric] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}
        rows.append(summary)
    return rows


def write_stage_a(study_root, code_commit, aa_floor, bwt_floor, task_il_floor):
    records = []
    for spec in stage_a_jobs():
        head_json = Path(study_root) / "heads" / spec.replace(":", "_") / "validation.json"
        records.append(audit_run(spec, find_run(spec, study_root), head_json, code_commit))
    promoted = select_stage_a(records, aa_floor, bwt_floor, task_il_floor)
    output = {
        "passed": bool(promoted),
        "promoted_candidates": promoted,
        "aa_final_floor": aa_floor,
        "bwt_floor": bwt_floor,
        "task_il_floor": task_il_floor,
        "frozen_head": FROZEN_HEAD,
        "selection_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "records": records,
    }
    _write_json(Path(study_root) / "STAGE_G_A_SUMMARY.json", output)
    return output


def write_stage_b(stage_a_summary, study_root, code_commit, aa_floor, bwt_floor, task_il_floor):
    stage_a = _read_json(stage_a_summary)
    promoted = stage_a.get("promoted_candidates", [])
    if not promoted:
        raise ValueError("Stage G B cannot run without promoted candidates")
    records = [row for row in stage_a["records"] if row["candidate"] in promoted]
    for spec in stage_b_jobs(stage_a_summary):
        head_json = Path(study_root) / "heads" / spec.replace(":", "_") / "validation.json"
        records.append(audit_run(spec, find_run(spec, study_root), head_json, code_commit))
    rows = summarize_rows(records)
    passing = [row for row in rows if eligible(
        {key: row[key]["mean"] for key in METRICS}, aa_floor, bwt_floor, task_il_floor
    )]
    passing.sort(key=lambda row: (
        row["aa_final_cil"]["mean"], row["task_il_final"]["mean"], row["bwt_cil"]["mean"]
    ), reverse=True)
    output = {
        "passed": bool(passing),
        "selected_candidate": passing[0]["candidate"] if passing else None,
        "selected_metrics": passing[0] if passing else None,
        "aa_final_floor": aa_floor,
        "bwt_floor": bwt_floor,
        "task_il_floor": task_il_floor,
        "frozen_head": FROZEN_HEAD,
        "selection_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "candidates": rows,
        "records": records,
    }
    _write_json(Path(study_root) / "STAGE_G_SELECTION.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("jobs-a")
    jobs_b = commands.add_parser("jobs-b")
    jobs_b.add_argument("--stage-a-summary", required=True)
    for name in ("claim-a", "claim-b"):
        child = commands.add_parser(name)
        child.add_argument("--claims-root", required=True)
        if name == "claim-b":
            child.add_argument("--stage-a-summary", required=True)
    command = commands.add_parser("command")
    command.add_argument("--job", required=True)
    command.add_argument("--study-root", required=True)
    command.add_argument("--repo-root", required=True)
    command.add_argument("--python", required=True)
    for name in ("find", "find-incomplete"):
        child = commands.add_parser(name)
        child.add_argument("--job", required=True)
        child.add_argument("--study-root", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--job", required=True)
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--head-json", required=True)
    audit.add_argument("--code-commit", required=True)
    for name in ("summarize-a", "summarize-b"):
        child = commands.add_parser(name)
        child.add_argument("--study-root", required=True)
        child.add_argument("--code-commit", required=True)
        child.add_argument("--aa-final-floor", type=float, default=0.37)
        child.add_argument("--bwt-floor", type=float, default=-0.18)
        child.add_argument("--task-il-floor", type=float, default=0.755)
        if name == "summarize-b":
            child.add_argument("--stage-a-summary", required=True)
    args = parser.parse_args()
    if args.action == "jobs-a":
        print("\n".join(stage_a_jobs()))
    elif args.action == "jobs-b":
        jobs = stage_b_jobs(args.stage_a_summary)
        if jobs:
            print("\n".join(jobs))
    elif args.action == "claim-a":
        job = claim_next(stage_a_jobs(), args.claims_root)
        if job:
            print(job)
    elif args.action == "claim-b":
        job = claim_next(stage_b_jobs(args.stage_a_summary), args.claims_root)
        if job:
            print(job)
    elif args.action == "command":
        print("\n".join(command_for(args.job, args.study_root, args.repo_root, args.python)))
    elif args.action in ("find", "find-incomplete"):
        print(find_run(args.job, args.study_root, args.action == "find"))
    elif args.action == "audit":
        print(json.dumps(audit_run(args.job, args.run_dir, args.head_json, args.code_commit), indent=2, sort_keys=True))
    elif args.action == "summarize-a":
        print(json.dumps(write_stage_a(
            args.study_root, args.code_commit, args.aa_final_floor, args.bwt_floor, args.task_il_floor
        ), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_stage_b(
            args.stage_a_summary, args.study_root, args.code_commit,
            args.aa_final_floor, args.bwt_floor, args.task_il_floor,
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
