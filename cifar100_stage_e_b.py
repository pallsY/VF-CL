#!/usr/bin/env python3
"""Three-seed validation for Stage E promoted balanced-replay candidates."""
import argparse
import json
import statistics
from pathlib import Path

from cifar100_stage_e_training import CANDIDATES, audit_run, find_run


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def promoted_candidates(stage_a_summary):
    summary = _read_json(stage_a_summary)
    promoted = summary.get("promoted_candidates") or []
    if (
        not summary.get("passed") or len(promoted) != 2
        or len(set(promoted)) != 2
        or any(candidate not in CANDIDATES for candidate in promoted)
    ):
        raise ValueError("Stage E A must promote two known candidates")
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


def write_summary(stage_a_summary, study_root, code_commit,
                  bwt_floor=-0.16):
    candidates = promoted_candidates(stage_a_summary)
    stage_a = _read_json(stage_a_summary)
    records = [
        record for record in stage_a["records"]
        if record["candidate"] in candidates and int(record["seed"]) == 42
    ]
    records.extend(
        audit_run(
            f"{candidate}:{seed}",
            find_run(f"{candidate}:{seed}", study_root),
            code_commit,
        )
        for candidate in candidates for seed in (43, 44)
    )
    if len({r["calibration_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Stage E B runs do not share calibration manifests")
    if len({r["validation_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Stage E B runs do not share validation manifests")
    rows = []
    for candidate in candidates:
        candidate_records = [
            record for record in records if record["candidate"] == candidate
        ]
        if {int(record["seed"]) for record in candidate_records} != {42, 43, 44}:
            raise ValueError(f"{candidate}: incomplete Stage E B seeds")
        row = {"candidate": candidate}
        for metric in ("aa_final_cil", "aa_avg_cil", "bwt_cil", "task_il_final"):
            values = [float(record["metrics"][metric]) for record in candidate_records]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values)
        row["eligible"] = row["bwt_cil_mean"] >= float(bwt_floor)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no Stage E B candidate satisfies the BWT floor")
    selected = max(
        eligible,
        key=lambda row: (
            row["aa_final_cil_mean"], row["task_il_final_mean"],
            row["bwt_cil_mean"], row["candidate"],
        ),
    )
    output = {
        "passed": True,
        "selected_candidate": selected["candidate"],
        "selected_metrics": selected,
        "bwt_floor": float(bwt_floor),
        "selection_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "candidates": rows,
        "records": records,
    }
    path = Path(study_root) / "STAGE_E_B_SELECTION.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    jobs = commands.add_parser("jobs")
    jobs.add_argument("--stage-a-summary", required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("--stage-a-summary", required=True)
    claim.add_argument("--claims-root", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--stage-a-summary", required=True)
    summary.add_argument("--study-root", required=True)
    summary.add_argument("--code-commit", required=True)
    summary.add_argument("--bwt-floor", type=float, default=-0.16)
    args = parser.parse_args()
    if args.action == "jobs":
        print("\n".join(job_specs(args.stage_a_summary)))
    elif args.action == "claim":
        job = claim_next(args.stage_a_summary, args.claims_root)
        if job: print(job)
    else:
        print(json.dumps(write_summary(
            args.stage_a_summary, args.study_root,
            args.code_commit, args.bwt_floor,
        ), indent=2, sort_keys=True))


if __name__ == "__main__": main()
