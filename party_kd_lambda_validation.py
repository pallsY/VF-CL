#!/usr/bin/env python3
"""CIFAR-100 PartyKD lambda validation matrix, audits, and selection."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from formal_cifar100_metrics import (
    METRICS,
    calibration_history,
    calibration_is_legal,
    compute_formal_metrics,
)


SEEDS = (42, 43, 44)
LAMBDAS = ("0.25", "0.50", "0.75", "1.00")
DATA_PATH = "/home/chase/Yangxx/VF-CL/data"


def parse_job(spec):
    method, seed_text = spec.split(":")
    seed = int(seed_text)
    valid_methods = {"lwf", *(f"lambda_{value}" for value in LAMBDAS)}
    if method not in valid_methods or seed not in SEEDS:
        raise ValueError(f"unknown job {spec}")
    return method, seed


def job_specs():
    return [
        *(f"lwf:{seed}" for seed in SEEDS),
        *(
            f"lambda_{value}:{seed}"
            for value in LAMBDAS
            for seed in SEEDS
        ),
    ]


def claim_next_job(claim_root):
    claim_root = Path(claim_root)
    claim_root.mkdir(parents=True, exist_ok=True)
    for spec in job_specs():
        try:
            (claim_root / spec.replace(":", "_")).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def expected_config(spec):
    method, seed = parse_job(spec)
    values = {
        "data": "cifar100",
        "data_path": DATA_PATH,
        "num_classes": 100,
        "num_tasks": 10,
        "classes_per_task": 10,
        "unlearn_after_tasks": [99],
        "unlearn_classes": [[0]],
        "num_parties": 4,
        "model_type": "resnet18",
        "aggregation": "sum",
        "epochs_per_task": 50,
        "batch_size": 64,
        "num_workers": 2,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "cl_method": "lwf" if method == "lwf" else "proto_evolve",
        "ul_method": "retrain",
        "deterministic": 1,
        "data_flow_audit": 1,
        "bic_enabled": 1,
        "bic_per_class": 25,
        "bic_split_seed": 20260722,
        "bic_lr": 0.05,
        "bic_steps": 1000,
        "bic_fit_mode": "joint_each_stage",
        "lambda_validation_enabled": 1,
        "lambda_validation_per_class": 25,
        "lambda_validation_split_seed": 20260729,
        "save_task_checkpoints": 3,
        "seed": seed,
        "device": "cuda:0",
        "replay_mode": "prototype",
    }
    if method == "lwf":
        values.update(
            lwf_temperature=2.0,
            lwf_lambda=1.0,
            lwf_ce_newonly=True,
            feat_distill_weight=0.0,
            dep_tracking_enabled=0,
            party_kd_enabled=0,
        )
    else:
        values.update(
            dep_tracking_enabled=1,
            party_kd_enabled=1,
            party_kd_mode="uniform",
            expected_party_kd_variant="uniform",
            party_kd_lambda=float(method.removeprefix("lambda_")),
        )
    return values


def command_for(spec, matrix_root, repo_root, python):
    method, seed = parse_job(spec)
    values = expected_config(spec)
    values["results_dir"] = str(Path(matrix_root) / "runs")
    label = method.replace(".", "p")
    values["exp_name"] = f"cifar100_lambda_validation_{label}_seed{seed}"
    command = [str(python), str(Path(repo_root) / "main.py")]
    for key, value in values.items():
        if key == "unlearn_after_tasks":
            cli_value = ",".join(str(item) for item in value)
        elif key == "unlearn_classes":
            cli_value = ";".join(
                ",".join(str(item) for item in group) for group in value
            )
        else:
            cli_value = str(value)
        command.extend((f"--{key}", cli_value))
    return command


def _normalized_config(config, expected):
    normalized = {}
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, bool):
            actual = bool(actual)
        elif isinstance(value, int):
            actual = int(actual)
        elif isinstance(value, float):
            actual = float(actual)
        elif not isinstance(value, list):
            actual = str(actual)
        normalized[key] = actual
    return normalized


def metrics_for_result(spec, result):
    method, _ = parse_job(spec)
    if method == "lwf":
        history = result.get("task_acc_history") or []
    else:
        events = result.get("bic_history") or []
        if len(events) != 10 or not calibration_is_legal(events):
            raise ValueError(f"{spec}: illegal calibrated validation trajectory")
        history = calibration_history(events)
    return compute_formal_metrics(history, expected_tasks=10)


def audit_run(spec, run_dir, code_commit):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    result_path = run_dir / "results.json"
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(f"{spec}: missing config or results")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{spec}: missing final checkpoint")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = expected_config(spec)
    actual = _normalized_config(config, expected)
    if actual != expected:
        mismatch = {
            key: (actual[key], expected[key])
            for key in expected
            if actual[key] != expected[key]
        }
        raise ValueError(f"{spec}: protocol mismatch {mismatch}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    calibration = result.get("calibration_audit", {})
    if not calibration.get("passed") or calibration.get("test_used_for_fit"):
        raise ValueError(f"{spec}: calibration audit failed")
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
        raise ValueError(f"{spec}: selection audit failed")
    for key in ("calibration_manifest_sha256", "validation_manifest_sha256"):
        if len(str(selection.get(key, ""))) != 64:
            raise ValueError(f"{spec}: invalid {key}")

    metrics = metrics_for_result(spec, result)
    protocol = {"job": spec, "code_commit": code_commit, "config": expected}
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    record = {
        **protocol,
        "protocol_sha256": hashlib.sha256(encoded).hexdigest(),
        "run_dir": str(run_dir.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "calibration_manifest_sha256": selection["calibration_manifest_sha256"],
        "validation_manifest_sha256": selection["validation_manifest_sha256"],
        "metrics": metrics,
    }
    (run_dir / "protocol_digest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def _run_prefix(spec):
    method, seed = parse_job(spec)
    return f"cifar100_lambda_validation_{method.replace('.', 'p')}_seed{seed}_"


def _matching_runs(spec, matrix_root):
    runs = Path(matrix_root) / "runs"
    return sorted(
        (path for path in runs.glob(_run_prefix(spec) + "*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_complete(spec, matrix_root):
    for run_dir in _matching_runs(spec, matrix_root):
        if (run_dir / "results.json").is_file():
            return run_dir
    raise FileNotFoundError(spec)


def find_incomplete(spec, matrix_root):
    for run_dir in _matching_runs(spec, matrix_root):
        if (
            (run_dir / "config.json").is_file()
            and not (run_dir / "results.json").is_file()
        ):
            return run_dir
    raise FileNotFoundError(spec)


def recorded_code_commit(spec, run_dir):
    record = json.loads(
        (Path(run_dir) / "protocol_digest.json").read_text(encoding="utf-8")
    )
    if record.get("job") != spec:
        raise ValueError(f"{spec}: protocol digest job mismatch")
    commit = record.get("code_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError(f"{spec}: protocol digest has no code commit")
    return commit


def aggregate_records(records):
    grouped = {}
    for record in records:
        method, seed = parse_job(record["job"])
        grouped.setdefault(method, {})[seed] = record["metrics"]
    rows = []
    for method, by_seed in sorted(grouped.items()):
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"{method}: incomplete seeds")
        row = {"method": method, "seeds": ",".join(map(str, SEEDS))}
        for metric in METRICS:
            values = [float(by_seed[seed][metric]) for seed in SEEDS]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values)
        rows.append(row)
    return rows


def choose_lambda(rows):
    lwf = [row for row in rows if row["method"] == "lwf"]
    if len(lwf) != 1:
        raise ValueError("validation summary must contain exactly one LwF row")
    constraint = float(lwf[0]["bwt_cil_mean"])
    eligible = [
        row for row in rows
        if row["method"].startswith("lambda_")
        and float(row["bwt_cil_mean"]) >= constraint
    ]
    if not eligible:
        raise ValueError("no lambda satisfies the LwF validation BWT constraint")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["aa_final_cil_mean"]),
            float(row["task_il_final_mean"]),
            float(row["bwt_cil_mean"]),
            -float(row["method"].removeprefix("lambda_")),
        ),
    )
    return {
        **selected,
        "lambda": float(selected["method"].removeprefix("lambda_")),
        "lwf_bwt_constraint": constraint,
        "eligible_methods": sorted(row["method"] for row in eligible),
    }


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select(matrix_root, output_dir, code_commit):
    records = [
        audit_run(spec, find_complete(spec, matrix_root), code_commit)
        for spec in job_specs()
    ]
    calibration_hashes = {item["calibration_manifest_sha256"] for item in records}
    validation_hashes = {item["validation_manifest_sha256"] for item in records}
    if len(calibration_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("validation runs do not share identical manifests")

    per_run = []
    for record in records:
        method, seed = parse_job(record["job"])
        per_run.append({"method": method, "seed": seed, **record["metrics"]})
    summary = aggregate_records(records)
    selected = choose_lambda(summary)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "VALIDATION_PER_RUN.csv", per_run)
    _write_csv(output / "VALIDATION_TABLE.csv", summary)
    audit = {
        "passed": True,
        "jobs": job_specs(),
        "code_commit": code_commit,
        "calibration_manifest_sha256": next(iter(calibration_hashes)),
        "validation_manifest_sha256": next(iter(validation_hashes)),
        "test_used_for_selection": False,
    }
    (output / "VALIDATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "VALIDATION_SELECTION.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SELECTED_LAMBDA").write_text(
        f"{selected['lambda']:.2f}\n", encoding="utf-8"
    )
    return selected


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("jobs")
    claim = subparsers.add_parser("claim")
    claim.add_argument("--claims-root", required=True)
    command = subparsers.add_parser("command")
    command.add_argument("--job", required=True)
    command.add_argument("--matrix-root", required=True)
    command.add_argument("--repo-root", required=True)
    command.add_argument("--python", required=True)
    for action in ("find", "find-incomplete"):
        child = subparsers.add_parser(action)
        child.add_argument("--job", required=True)
        child.add_argument("--matrix-root", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--job", required=True)
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--code-commit", required=True)
    provenance = subparsers.add_parser("provenance")
    provenance.add_argument("--job", required=True)
    provenance.add_argument("--run-dir", required=True)
    selection = subparsers.add_parser("select")
    selection.add_argument("--matrix-root", required=True)
    selection.add_argument("--output-dir", required=True)
    selection.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    if args.action == "jobs":
        print("\n".join(job_specs()))
    elif args.action == "claim":
        job = claim_next_job(args.claims_root)
        if job is None:
            raise SystemExit(1)
        print(job)
    elif args.action == "command":
        print("\n".join(command_for(
            args.job, args.matrix_root, args.repo_root, args.python
        )))
    elif args.action == "find":
        print(find_complete(args.job, args.matrix_root))
    elif args.action == "find-incomplete":
        print(find_incomplete(args.job, args.matrix_root))
    elif args.action == "audit":
        print(json.dumps(
            audit_run(args.job, args.run_dir, args.code_commit),
            sort_keys=True,
        ))
    elif args.action == "provenance":
        print(recorded_code_commit(args.job, args.run_dir))
    elif args.action == "select":
        print(json.dumps(
            select(args.matrix_root, args.output_dir, args.code_commit),
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
