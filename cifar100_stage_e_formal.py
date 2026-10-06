#!/usr/bin/env python3
"""Formal CIFAR-100 reruns for the validation-selected Stage E candidate."""
import argparse
import hashlib
import json
from pathlib import Path

from cifar100_stage_e_training import CANDIDATES, expected_config
from formal_cifar100_metrics import (
    calibration_history,
    calibration_is_legal,
    compute_formal_metrics,
)


SEEDS = (42, 43, 44)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def selected_candidate(selection_path):
    selection = _read_json(selection_path)
    candidate = selection.get("selected_candidate")
    if (
        not selection.get("passed")
        or selection.get("test_used_for_selection")
        or selection.get("selection_source") != "cifar100-train-validation"
        or candidate not in CANDIDATES
    ):
        raise ValueError("formal rerun requires a legal Stage E selection")
    return candidate


def parse_job(spec, selection_path):
    candidate, seed = spec.rsplit(":", 1)
    if candidate != selected_candidate(selection_path) or int(seed) not in SEEDS:
        raise ValueError(f"unknown Stage E formal job: {spec}")
    return candidate, int(seed)


def job_specs(selection_path):
    candidate = selected_candidate(selection_path)
    return [f"{candidate}:{seed}" for seed in SEEDS]


def expected_formal_config(spec, selection_path):
    candidate, seed = parse_job(spec, selection_path)
    config = expected_config(f"{candidate}:{seed}")
    config["lambda_validation_enabled"] = 0
    return config


def command_for(spec, selection_path, study_root, repo_root, python):
    candidate, seed = parse_job(spec, selection_path)
    values = expected_formal_config(spec, selection_path)
    values["results_dir"] = str(Path(study_root) / "runs")
    values["exp_name"] = (
        "cifar100_stage_e_formal_"
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


def _prefix(spec, selection_path):
    candidate, seed = parse_job(spec, selection_path)
    return (
        "cifar100_stage_e_formal_"
        + candidate.replace(".", "p") + f"_seed{seed}_"
    )


def find_run(spec, selection_path, study_root, complete=True):
    paths = sorted(
        (Path(study_root) / "runs").glob(
            _prefix(spec, selection_path) + "*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in paths:
        config = (run_dir / "config.json").is_file()
        result = (run_dir / "results.json").is_file()
        if (complete and result) or (not complete and config and not result):
            return run_dir
    raise FileNotFoundError(spec)


def claim_next(selection_path, claims_root):
    claims_root = Path(claims_root)
    claims_root.mkdir(parents=True, exist_ok=True)
    for spec in job_specs(selection_path):
        try:
            (claims_root / spec.replace(":", "_")).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def audit_run(spec, selection_path, run_dir, code_commit):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    result_path = run_dir / "results.json"
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(f"{spec}: missing formal config or results")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{spec}: missing formal final checkpoint")
    expected = expected_formal_config(spec, selection_path)
    actual = _normalized(_read_json(config_path), expected)
    if actual != expected:
        mismatch = {
            key: (actual[key], expected[key])
            for key in expected if actual[key] != expected[key]
        }
        raise ValueError(f"{spec}: formal protocol mismatch {mismatch}")
    result = _read_json(result_path)
    calibration = result.get("calibration_audit", {})
    if not calibration.get("passed") or calibration.get("test_used_for_fit"):
        raise ValueError(f"{spec}: formal calibration audit failed")
    events = result.get("bic_history") or []
    if len(events) != 10 or not calibration_is_legal(events):
        raise ValueError(f"{spec}: illegal formal calibrated trajectory")
    metrics = compute_formal_metrics(calibration_history(events), expected_tasks=10)
    candidate, seed = parse_job(spec, selection_path)
    record = {
        "job": spec, "candidate": candidate, "seed": seed,
        "code_commit": code_commit, "config": expected,
        "run_dir": str(run_dir.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "metrics_before_balanced_head": metrics,
        "calibration_manifest_sha256": calibration["manifest_sha256"],
        "selection_sha256": hashlib.sha256(
            Path(selection_path).read_bytes()
        ).hexdigest(),
        "test_used_for_selection": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    jobs = commands.add_parser("jobs")
    jobs.add_argument("--selection", required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("--selection", required=True)
    claim.add_argument("--claims-root", required=True)
    command = commands.add_parser("command")
    command.add_argument("--job", required=True)
    command.add_argument("--selection", required=True)
    command.add_argument("--study-root", required=True)
    command.add_argument("--repo-root", required=True)
    command.add_argument("--python", required=True)
    for name in ("find", "find-incomplete"):
        child = commands.add_parser(name)
        child.add_argument("--job", required=True)
        child.add_argument("--selection", required=True)
        child.add_argument("--study-root", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--job", required=True)
    audit.add_argument("--selection", required=True)
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.action == "jobs": print("\n".join(job_specs(args.selection)))
    elif args.action == "claim":
        job = claim_next(args.selection, args.claims_root)
        if job: print(job)
    elif args.action == "command":
        print("\n".join(command_for(
            args.job, args.selection, args.study_root, args.repo_root, args.python
        )))
    elif args.action in ("find", "find-incomplete"):
        print(find_run(
            args.job, args.selection, args.study_root,
            complete=args.action == "find",
        ))
    else:
        print(json.dumps(audit_run(
            args.job, args.selection, args.run_dir, args.code_commit
        ), indent=2, sort_keys=True))


if __name__ == "__main__": main()
