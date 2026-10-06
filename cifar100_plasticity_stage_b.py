#!/usr/bin/env python3
"""Three-seed validation for candidates promoted by plasticity Stage A."""
import argparse
import json
import statistics
from pathlib import Path

from cifar100_plasticity_validation import (
    audit_run,
    command_for,
    find_run,
)


SEEDS = (42, 43, 44)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def promoted_candidates(stage_a_summary):
    summary = _read_json(stage_a_summary)
    promoted = summary.get("promoted_candidates") or []
    if not summary.get("passed") or len(promoted) != 2 or len(set(promoted)) != 2:
        raise ValueError("Stage A must promote exactly two candidates")
    return promoted


def job_specs(stage_a_summary):
    return [
        f"{candidate}:{seed}"
        for candidate in promoted_candidates(stage_a_summary)
        for seed in (43, 44)
    ]


def claim_next(stage_a_summary, claims_root):
    claims_root = Path(claims_root)
    claims_root.mkdir(parents=True, exist_ok=True)
    for spec in job_specs(stage_a_summary):
        try:
            (claims_root / spec.replace(":", "_")).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def _stage_a_records(stage_a_summary, candidates):
    summary = _read_json(stage_a_summary)
    by_candidate = {
        record["candidate"]: record
        for record in summary.get("records") or []
        if int(record.get("seed", -1)) == 42
    }
    if any(candidate not in by_candidate for candidate in candidates):
        raise ValueError("Stage A summary is missing a promoted seed-42 record")
    return [by_candidate[candidate] for candidate in candidates]


def write_summary(stage_a_summary, study_root, code_commit, bwt_floor=-0.20):
    candidates = promoted_candidates(stage_a_summary)
    records = _stage_a_records(stage_a_summary, candidates)
    records.extend(
        audit_run(
            f"{candidate}:{seed}",
            find_run(f"{candidate}:{seed}", study_root),
            code_commit,
        )
        for candidate in candidates
        for seed in (43, 44)
    )
    calibration_hashes = {item["calibration_manifest_sha256"] for item in records}
    validation_hashes = {item["validation_manifest_sha256"] for item in records}
    if len(calibration_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("Stage B runs do not share validation manifests")
    rows = []
    for candidate in candidates:
        candidate_records = [
            record for record in records if record["candidate"] == candidate
        ]
        if {int(record["seed"]) for record in candidate_records} != set(SEEDS):
            raise ValueError(f"{candidate}: incomplete Stage B seeds")
        row = {"candidate": candidate}
        for metric in ("aa_final_cil", "aa_avg_cil", "bwt_cil", "task_il_final"):
            values = [float(record["metrics"][metric]) for record in candidate_records]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values)
        row["eligible"] = row["bwt_cil_mean"] >= float(bwt_floor)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no Stage B candidate satisfies the BWT floor")
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
        "selected_metrics": selected,
        "bwt_floor": float(bwt_floor),
        "test_used_for_selection": False,
        "selection_source": "cifar100-train-validation",
        "candidates": rows,
        "records": records,
    }
    path = Path(study_root) / "STAGE_B_SELECTION.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return output


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("--stage-a-summary", required=True)
    claim = subparsers.add_parser("claim")
    claim.add_argument("--stage-a-summary", required=True)
    claim.add_argument("--claims-root", required=True)
    command = subparsers.add_parser("command")
    command.add_argument("--job", required=True)
    command.add_argument("--study-root", required=True)
    command.add_argument("--repo-root", required=True)
    command.add_argument("--python", required=True)
    for name in ("find", "find-incomplete"):
        child = subparsers.add_parser(name)
        child.add_argument("--job", required=True)
        child.add_argument("--study-root", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--job", required=True)
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--code-commit", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--stage-a-summary", required=True)
    summary.add_argument("--study-root", required=True)
    summary.add_argument("--code-commit", required=True)
    summary.add_argument("--bwt-floor", type=float, default=-0.20)
    args = parser.parse_args()
    if args.action == "jobs":
        print("\n".join(job_specs(args.stage_a_summary)))
    elif args.action == "claim":
        job = claim_next(args.stage_a_summary, args.claims_root)
        if job:
            print(job)
    elif args.action == "command":
        print("\n".join(command_for(
            args.job, args.study_root, args.repo_root, args.python
        )))
    elif args.action == "find":
        print(find_run(args.job, args.study_root))
    elif args.action == "find-incomplete":
        print(find_run(args.job, args.study_root, complete=False))
    elif args.action == "audit":
        print(json.dumps(audit_run(
            args.job, args.run_dir, args.code_commit
        ), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_summary(
            args.stage_a_summary, args.study_root,
            args.code_commit, args.bwt_floor,
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
