#!/usr/bin/env python3
"""Stage E stable balanced-prototype replay validation sweep."""
import argparse
import hashlib
import json
from pathlib import Path

from feature_retention_validation import expected_validation_config
from formal_cifar100_metrics import (
    calibration_history,
    calibration_is_legal,
    compute_formal_metrics,
)


CANDIDATES = {
    "mean_l0.10_m0.00": {"proto_lambda_a": 0.10, "fim_freeze_frac": 0.00},
    "mean_l0.15_m0.00": {"proto_lambda_a": 0.15, "fim_freeze_frac": 0.00},
    "mean_l0.10_m0.05": {"proto_lambda_a": 0.10, "fim_freeze_frac": 0.05},
    "mean_l0.15_m0.05": {"proto_lambda_a": 0.15, "fim_freeze_frac": 0.05},
}


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_job(spec):
    candidate, seed = spec.rsplit(":", 1)
    if candidate not in CANDIDATES or int(seed) not in (42, 43, 44):
        raise ValueError(f"unknown Stage E job: {spec}")
    return candidate, int(seed)


def job_specs(seed=42):
    return [f"{candidate}:{int(seed)}" for candidate in CANDIDATES]


def expected_config(spec):
    candidate, seed = parse_job(spec)
    config = expected_validation_config(f"feat_0.05:{seed}")
    config.update({
        "distill_weight": 0.25,
        "feat_distill_weight": 0.05,
        "proto_replay_loss_norm": "sample_mean",
        "proto_replay_ratio": 1.0,
        **CANDIDATES[candidate],
    })
    return config


def command_for(spec, study_root, repo_root, python):
    candidate, seed = parse_job(spec)
    values = expected_config(spec)
    values["results_dir"] = str(Path(study_root) / "runs")
    values["exp_name"] = (
        "cifar100_stage_e_validation_"
        + candidate.replace(".", "p") + f"_seed{seed}"
    )
    command = [str(python), str(Path(repo_root) / "main.py")]
    for key, value in values.items():
        if key == "unlearn_after_tasks":
            cli_value = ",".join(str(item) for item in value)
        elif key == "unlearn_classes":
            cli_value = ";".join(
                ",".join(map(str, group)) for group in value
            )
        else:
            cli_value = str(value)
        command.extend((f"--{key}", cli_value))
    return command


def _normalized(config, expected):
    output = {}
    for key, expected_value in expected.items():
        actual = config.get(key)
        if isinstance(expected_value, bool):
            actual = bool(actual)
        elif isinstance(expected_value, int):
            actual = int(actual)
        elif isinstance(expected_value, float):
            actual = float(actual)
        elif not isinstance(expected_value, list):
            actual = str(actual)
        output[key] = actual
    return output


def _prefix(spec):
    candidate, seed = parse_job(spec)
    return (
        "cifar100_stage_e_validation_"
        + candidate.replace(".", "p") + f"_seed{seed}_"
    )


def find_run(spec, study_root, complete=True):
    paths = sorted(
        (Path(study_root) / "runs").glob(_prefix(spec) + "*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in paths:
        config = (run_dir / "config.json").is_file()
        result = (run_dir / "results.json").is_file()
        if (complete and result) or (not complete and config and not result):
            return run_dir
    raise FileNotFoundError(spec)


def claim_next(claims_root):
    claims_root = Path(claims_root)
    claims_root.mkdir(parents=True, exist_ok=True)
    for spec in job_specs():
        try:
            (claims_root / spec.replace(":", "_")).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def audit_run(spec, run_dir, code_commit):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    result_path = run_dir / "results.json"
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(f"{spec}: missing config or results")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{spec}: missing final checkpoint")
    expected = expected_config(spec)
    actual = _normalized(_read_json(config_path), expected)
    if actual != expected:
        mismatch = {
            key: (actual[key], expected[key])
            for key in expected if actual[key] != expected[key]
        }
        raise ValueError(f"{spec}: protocol mismatch {mismatch}")
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
    metrics = compute_formal_metrics(calibration_history(events), expected_tasks=10)
    candidate, seed = parse_job(spec)
    record = {
        "job": spec, "candidate": candidate, "seed": seed,
        "code_commit": code_commit, "config": expected,
        "run_dir": str(run_dir.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str(checkpoint.resolve()), "metrics": metrics,
        "calibration_manifest_sha256": selection["calibration_manifest_sha256"],
        "validation_manifest_sha256": selection["validation_manifest_sha256"],
        "test_used_for_selection": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def write_summary(study_root, code_commit, bwt_floor=-0.16):
    records = [
        audit_run(spec, find_run(spec, study_root), code_commit)
        for spec in job_specs()
    ]
    if len({r["calibration_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Stage E runs do not share calibration manifests")
    if len({r["validation_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Stage E runs do not share validation manifests")
    eligible = [
        record for record in records
        if float(record["metrics"]["bwt_cil"]) >= float(bwt_floor)
    ]
    promoted = sorted(
        eligible,
        key=lambda item: (
            float(item["metrics"]["aa_final_cil"]),
            float(item["metrics"]["task_il_final"]),
            float(item["metrics"]["bwt_cil"]),
        ), reverse=True,
    )[:2]
    output = {
        "passed": len(promoted) == 2,
        "bwt_floor": float(bwt_floor),
        "promoted_candidates": [item["candidate"] for item in promoted],
        "selection_source": "cifar100-train-validation",
        "test_used_for_selection": False,
        "records": records,
    }
    path = Path(study_root) / "STAGE_E_A_SUMMARY.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return output


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("jobs")
    claim = commands.add_parser("claim")
    claim.add_argument("--claims-root", required=True)
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
    audit.add_argument("--code-commit", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--study-root", required=True)
    summary.add_argument("--code-commit", required=True)
    summary.add_argument("--bwt-floor", type=float, default=-0.16)
    args = parser.parse_args()
    if args.action == "jobs": print("\n".join(job_specs()))
    elif args.action == "claim":
        job = claim_next(args.claims_root)
        if job: print(job)
    elif args.action == "command":
        print("\n".join(command_for(
            args.job, args.study_root, args.repo_root, args.python
        )))
    elif args.action in ("find", "find-incomplete"):
        print(find_run(args.job, args.study_root, args.action == "find"))
    elif args.action == "audit":
        print(json.dumps(audit_run(
            args.job, args.run_dir, args.code_commit
        ), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_summary(
            args.study_root, args.code_commit, args.bwt_floor
        ), indent=2, sort_keys=True))


if __name__ == "__main__": main()
