#!/usr/bin/env python3
"""Staged CIFAR-100 feature-retention validation and formal reporting."""

import argparse
import hashlib
import csv
import json
import os
import statistics
from pathlib import Path

from formal_cifar100_metrics import (
    METRICS,
    aggregate_rows,
    calibration_history,
    calibration_is_legal,
    collect_external_runs,
    compute_formal_metrics,
    write_outputs,
)
from party_kd_lambda_validation import (
    expected_config as lambda_expected_config,
    find_complete,
    metrics_for_result,
    parse_job as parse_reference_job,
)


WEIGHTS = ("0.05", "0.20", "1.00")
SEEDS = (42, 43, 44)


def parse_job(spec):
    try:
        label, seed_text = spec.split(":")
        weight = label.removeprefix("feat_")
        seed = int(seed_text)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"unknown job {spec}") from None
    if not label.startswith("feat_") or weight not in WEIGHTS or seed not in SEEDS:
        raise ValueError(f"unknown job {spec}")
    return weight, seed


def validation_job_specs(weights=WEIGHTS, seeds=SEEDS):
    return [f"feat_{weight}:{seed}" for weight in weights for seed in seeds]


def _read_weights(path, maximum):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing selection file {path}")
    weights = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not weights or len(weights) > maximum or len(set(weights)) != len(weights):
        raise ValueError(f"invalid selection file {path}")
    if any(weight not in WEIGHTS for weight in weights):
        raise ValueError(f"unknown selected weight in {path}")
    return weights


def stage_job_specs(stage, selection_dir):
    selection_dir = Path(selection_dir)
    if stage == "A":
        return validation_job_specs(WEIGHTS, (42,))
    if stage == "B":
        weights = _read_weights(selection_dir / "PROMOTED_WEIGHTS", maximum=2)
        return validation_job_specs(tuple(weights), (43, 44))
    if stage == "formal":
        weight = _read_weights(
            selection_dir / "SELECTED_FEAT_DISTILL_WEIGHT", maximum=1
        )[0]
        return validation_job_specs((weight,), SEEDS)
    raise ValueError(f"unknown stage {stage}")


def claim_next_job(jobs, claim_root):
    claim_root = Path(claim_root)
    claim_root.mkdir(parents=True, exist_ok=True)
    for spec in jobs:
        try:
            (claim_root / spec.replace(":", "_")).mkdir()
        except FileExistsError:
            continue
        return spec
    return None


def expected_validation_config(spec):
    weight, seed = parse_job(spec)
    config = lambda_expected_config(f"lambda_1.00:{seed}")
    config["feat_distill_weight"] = float(weight)
    return config


def expected_formal_config(weight, seed):
    config = expected_validation_config(f"feat_{weight}:{seed}")
    config["lambda_validation_enabled"] = 0
    return config


def command_for(spec, mode, study_root, repo_root, python):
    weight, seed = parse_job(spec)
    if mode == "validation":
        values = expected_validation_config(spec)
    elif mode == "formal":
        values = expected_formal_config(weight, seed)
    else:
        raise ValueError(f"unknown mode {mode}")
    values["results_dir"] = str(Path(study_root) / mode / "runs")
    tag = weight.replace(".", "p")
    values["exp_name"] = (
        f"cifar100_feature_retention_{mode}_feat_{tag}_seed{seed}"
    )
    command = [str(python), str(Path(repo_root) / "main.py")]
    for key, value in values.items():
        if key == "unlearn_after_tasks":
            cli_value = ",".join(str(item) for item in value)
        elif key == "unlearn_classes":
            cli_value = ";".join(",".join(map(str, group)) for group in value)
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


def audit_candidate_run(spec, run_dir, code_commit, mode="validation"):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    result_path = run_dir / "results.json"
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    if not config_path.is_file() or not result_path.is_file():
        raise ValueError(f"{spec}: missing config or results")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{spec}: missing final checkpoint")

    weight, seed = parse_job(spec)
    expected = (
        expected_validation_config(spec)
        if mode == "validation"
        else expected_formal_config(weight, seed)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actual = _normalized(config, expected)
    if actual != expected:
        mismatch = {
            key: (actual[key], expected[key])
            for key in expected
            if actual[key] != expected[key]
        }
        label = "formal protocol mismatch" if mode == "formal" else "protocol mismatch"
        raise ValueError(f"{spec}: {label} {mismatch}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = {}
    if mode == "validation":
        selection = result.get("selection_audit", {})
        overlaps = (
            "training_calibration_overlap_count",
            "training_validation_overlap_count",
            "calibration_validation_overlap_count",
        )
        if (
            not selection.get("passed")
            or selection.get("test_used_for_selection")
            or selection.get("evaluation_source") != "cifar100-train-validation"
            or any(int(selection.get(key, -1)) != 0 for key in overlaps)
        ):
            raise ValueError(f"{spec}: selection audit failed")
        for key in ("calibration_manifest_sha256", "validation_manifest_sha256"):
            if len(str(selection.get(key, ""))) != 64:
                raise ValueError(f"{spec}: invalid {key}")

    events = result.get("bic_history") or []
    if len(events) != 10 or not calibration_is_legal(events):
        raise ValueError(f"{spec}: illegal calibrated trajectory")
    metrics = compute_formal_metrics(calibration_history(events), expected_tasks=10)
    record = {
        "job": spec,
        "weight": weight,
        "seed": seed,
        "mode": mode,
        "code_commit": code_commit,
        "run_dir": str(run_dir.resolve()),
        "result": str(result_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "config": expected,
        "metrics": metrics,
    }
    if mode == "validation":
        record.update(
            calibration_manifest_sha256=selection["calibration_manifest_sha256"],
            validation_manifest_sha256=selection["validation_manifest_sha256"],
        )
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def load_reference_record(spec, reference_root):
    run_dir = find_complete(spec, reference_root)
    digest_path = run_dir / "protocol_digest.json"
    if not digest_path.is_file():
        raise ValueError(f"{spec}: missing reference protocol digest")
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    if digest.get("job") != spec or digest.get("config") != lambda_expected_config(spec):
        raise ValueError(f"{spec}: reference protocol mismatch")
    result_path = run_dir / "results.json"
    checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
    if not result_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"{spec}: incomplete reference artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed = metrics_for_result(spec, result)
    if recomputed != digest.get("metrics"):
        raise ValueError(f"{spec}: reference metric mismatch")
    method, seed = parse_reference_job(spec)
    return {
        "job": spec,
        "weight": "0.00" if method == "lambda_1.00" else "lwf",
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "metrics": recomputed,
        "calibration_manifest_sha256": digest["calibration_manifest_sha256"],
        "validation_manifest_sha256": digest["validation_manifest_sha256"],
        "code_commit": digest["code_commit"],
    }


def _require_shared_manifests(records):
    calibration = {record["calibration_manifest_sha256"] for record in records}
    validation = {record["validation_manifest_sha256"] for record in records}
    if len(calibration) != 1 or len(validation) != 1:
        raise ValueError("validation records do not share identical manifests")


def _eligible(candidate, baseline, lwf):
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    lwf_metrics = lwf["metrics"]
    return (
        float(candidate_metrics["aa_final_cil"])
        > float(baseline_metrics["aa_final_cil"])
        and float(candidate_metrics["task_il_final"])
        >= float(baseline_metrics["task_il_final"])
        and float(candidate_metrics["bwt_cil"])
        >= float(lwf_metrics["bwt_cil"])
    )


def select_stage_a(records, baseline, lwf):
    _require_shared_manifests([*records, baseline, lwf])
    eligible = [record for record in records if _eligible(record, baseline, lwf)]
    eligible.sort(
        key=lambda record: (
            float(record["metrics"]["aa_final_cil"]),
            float(record["metrics"]["task_il_final"]),
            float(record["metrics"]["bwt_cil"]),
            -float(record["weight"]),
        ),
        reverse=True,
    )
    return [record["weight"] for record in eligible[:2]]


def _aggregate(records):
    by_weight = {}
    for record in records:
        by_weight.setdefault(record["weight"], {})[record["seed"]] = record
    output = {}
    for weight, by_seed in by_weight.items():
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"feat_{weight}: incomplete seeds")
        output[weight] = {
            metric: statistics.fmean(
                float(by_seed[seed]["metrics"][metric]) for seed in SEEDS
            )
            for metric in METRICS
        }
    return output


def select_stage_b(records, baselines, lwfs):
    _require_shared_manifests([*records, *baselines, *lwfs])
    candidates = _aggregate(records)
    baseline = _aggregate(baselines)["0.00"]
    lwf = _aggregate(lwfs)["lwf"]
    eligible = [
        (weight, metrics)
        for weight, metrics in candidates.items()
        if metrics["aa_final_cil"] > baseline["aa_final_cil"]
        and metrics["task_il_final"] >= baseline["task_il_final"]
        and metrics["bwt_cil"] >= lwf["bwt_cil"]
    ]
    if not eligible:
        return None
    weight, metrics = max(
        eligible,
        key=lambda item: (
            item[1]["aa_final_cil"],
            item[1]["task_il_final"],
            item[1]["bwt_cil"],
            -float(item[0]),
        ),
    )
    return {"weight": weight, "metrics": metrics}


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_records_csv(path, records):
    rows = [
        {
            "weight": record["weight"],
            "seed": record["seed"],
            **record["metrics"],
        }
        for record in records
    ]
    fields = ["weight", "seed", *METRICS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_stage_a_selection(records, baseline, lwf, study_root):
    study_root = Path(study_root)
    selection = study_root / "selection"
    promoted = select_stage_a(records, baseline, lwf)
    _write_records_csv(selection / "STAGE_A_PER_RUN.csv", records)
    _atomic_write(
        selection / "STAGE_A_SELECTION.json",
        json.dumps({
            "passed": bool(promoted),
            "promoted_weights": promoted,
            "test_used_for_selection": False,
        }, indent=2, sort_keys=True) + "\n",
    )
    if promoted:
        _atomic_write(selection / "PROMOTED_WEIGHTS", "\n".join(promoted) + "\n")
    else:
        _atomic_write(study_root / "FEATURE_RETENTION_NO_CANDIDATE", "stage_a\n")
    return promoted


def write_stage_b_selection(records, baselines, lwfs, study_root):
    study_root = Path(study_root)
    selection = study_root / "selection"
    selected = select_stage_b(records, baselines, lwfs)
    _write_records_csv(selection / "STAGE_B_PER_RUN.csv", records)
    _atomic_write(
        selection / "STAGE_B_SELECTION.json",
        json.dumps({
            "passed": selected is not None,
            "selected": selected,
            "test_used_for_selection": False,
        }, indent=2, sort_keys=True) + "\n",
    )
    if selected is None:
        _atomic_write(study_root / "FEATURE_RETENTION_NO_CANDIDATE", "stage_b\n")
    else:
        _atomic_write(
            selection / "SELECTED_FEAT_DISTILL_WEIGHT",
            selected["weight"] + "\n",
        )
    return selected


REFERENCE_METHODS = {
    "LwF",
    "Ours (Uniform KD)",
    "Ours (Uniform KD + Joint Cal.)",
}


def audit_formal_run(spec, run_dir, code_commit):
    return audit_candidate_run(spec, run_dir, code_commit, mode="formal")


def build_formal_rows(reference_rows, candidate_records, weight):
    rows = [row for row in reference_rows if row["method"] in REFERENCE_METHODS]
    candidate_method = f"Ours (Uniform KD + feat-KD={weight} + Joint Cal.)"
    for record in candidate_records:
        if record["weight"] != weight:
            raise ValueError("formal candidate weight does not match selection")
        rows.append({
            "method": candidate_method,
            "seed": int(record["seed"]),
            **record["metrics"],
            "legacy_bwt": None,
            "calibration_legal": True,
            "source": record["result"],
        })
    for method in {*REFERENCE_METHODS, candidate_method}:
        seeds = [row["seed"] for row in rows if row["method"] == method]
        if sorted(seeds) != list(SEEDS):
            raise ValueError(f"{method}: missing or duplicate formal seeds")
    return sorted(rows, key=lambda row: (row["method"], row["seed"]))


def write_formal_comparison(external_root, candidate_records, weight, output_dir):
    reference_rows, reference_audit = collect_external_runs(external_root)
    rows = build_formal_rows(reference_rows, candidate_records, weight)
    summary = aggregate_rows(rows, expected_seeds=SEEDS)
    audit = {
        "passed": True,
        "expected_seeds": list(SEEDS),
        "selected_feat_distill_weight": float(weight),
        "selection_test_used": False,
        "reference_audit": reference_audit,
        "candidate_runs": [record["run_dir"] for record in candidate_records],
        "missing_by_method": {
            row["method"]: row["missing_seeds"] for row in summary
        },
    }
    write_outputs(output_dir, rows, summary, audit)
    output = Path(output_dir)
    for old_name, new_name in (
        ("FORMAL_CIFAR100_PER_RUN.csv", "FORMAL_FEATURE_RETENTION_PER_RUN.csv"),
        ("FORMAL_CIFAR100_TABLE.csv", "FORMAL_FEATURE_RETENTION_TABLE.csv"),
        ("FORMAL_CIFAR100_AUDIT.json", "FORMAL_FEATURE_RETENTION_AUDIT.json"),
        ("FORMAL_CIFAR100_TABLE.md", "FORMAL_FEATURE_RETENTION_TABLE.md"),
    ):
        (output / old_name).replace(output / new_name)
    return summary


def _run_prefix(spec, mode):
    weight, seed = parse_job(spec)
    return (
        f"cifar100_feature_retention_{mode}_"
        f"feat_{weight.replace('.', 'p')}_seed{seed}_"
    )


def _matching_runs(spec, mode, study_root):
    runs = Path(study_root) / mode / "runs"
    return sorted(
        (path for path in runs.glob(_run_prefix(spec, mode) + "*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_candidate_run(spec, mode, study_root, complete=True):
    for run_dir in _matching_runs(spec, mode, study_root):
        has_config = (run_dir / "config.json").is_file()
        has_result = (run_dir / "results.json").is_file()
        if (complete and has_result) or (not complete and has_config and not has_result):
            return run_dir
    raise FileNotFoundError(spec)


def _candidate_records(weights, seeds, study_root, code_commit, mode="validation"):
    return [
        audit_candidate_run(
            spec,
            find_candidate_run(spec, mode, study_root),
            code_commit,
            mode=mode,
        )
        for spec in validation_job_specs(tuple(weights), tuple(seeds))
    ]


def run_stage_a_selection(study_root, reference_root, code_commit):
    records = _candidate_records(WEIGHTS, (42,), study_root, code_commit)
    baseline = load_reference_record("lambda_1.00:42", reference_root)
    lwf = load_reference_record("lwf:42", reference_root)
    return write_stage_a_selection(records, baseline, lwf, study_root)


def run_stage_b_selection(study_root, reference_root, code_commit):
    selection = Path(study_root) / "selection"
    weights = _read_weights(selection / "PROMOTED_WEIGHTS", maximum=2)
    records = _candidate_records(weights, SEEDS, study_root, code_commit)
    baselines = [
        load_reference_record(f"lambda_1.00:{seed}", reference_root)
        for seed in SEEDS
    ]
    lwfs = [
        load_reference_record(f"lwf:{seed}", reference_root) for seed in SEEDS
    ]
    return write_stage_b_selection(records, baselines, lwfs, study_root)


def run_formal_report(study_root, external_root, code_commit):
    selection = Path(study_root) / "selection"
    weight = _read_weights(
        selection / "SELECTED_FEAT_DISTILL_WEIGHT", maximum=1
    )[0]
    records = _candidate_records(
        (weight,), SEEDS, study_root, code_commit, mode="formal"
    )
    return write_formal_comparison(
        external_root, records, weight, Path(study_root) / "formal_report"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("--stage", choices=("A", "B", "formal"), required=True)
    jobs.add_argument("--selection-dir", required=True)

    claim = subparsers.add_parser("claim")
    claim.add_argument("--stage", choices=("A", "B", "formal"), required=True)
    claim.add_argument("--claims-root", required=True)
    claim.add_argument("--selection-dir", required=True)

    command = subparsers.add_parser("command")
    command.add_argument("--job", required=True)
    command.add_argument("--mode", choices=("validation", "formal"), required=True)
    command.add_argument("--study-root", required=True)
    command.add_argument("--repo-root", required=True)
    command.add_argument("--python", required=True)

    for action in ("find", "find-incomplete"):
        child = subparsers.add_parser(action)
        child.add_argument("--job", required=True)
        child.add_argument("--mode", choices=("validation", "formal"), required=True)
        child.add_argument("--study-root", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--job", required=True)
    audit.add_argument("--mode", choices=("validation", "formal"), required=True)
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--code-commit", required=True)

    for action in ("select-stage-a", "select-stage-b"):
        child = subparsers.add_parser(action)
        child.add_argument("--study-root", required=True)
        child.add_argument("--reference-root", required=True)
        child.add_argument("--code-commit", required=True)

    report = subparsers.add_parser("report-formal")
    report.add_argument("--study-root", required=True)
    report.add_argument("--external-root", required=True)
    report.add_argument("--code-commit", required=True)

    args = parser.parse_args()
    if args.action == "jobs":
        print("\n".join(stage_job_specs(args.stage, args.selection_dir)))
    elif args.action == "claim":
        job = claim_next_job(
            stage_job_specs(args.stage, args.selection_dir), args.claims_root
        )
        if job is None:
            raise SystemExit(1)
        print(job)
    elif args.action == "command":
        print("\n".join(command_for(
            args.job, args.mode, args.study_root, args.repo_root, args.python
        )))
    elif args.action in ("find", "find-incomplete"):
        print(find_candidate_run(
            args.job,
            args.mode,
            args.study_root,
            complete=args.action == "find",
        ))
    elif args.action == "audit":
        record = audit_candidate_run(
            args.job, args.run_dir, args.code_commit, mode=args.mode
        )
        _atomic_write(
            Path(args.run_dir) / "feature_retention_protocol_digest.json",
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(record, sort_keys=True))
    elif args.action == "select-stage-a":
        print(json.dumps(run_stage_a_selection(
            args.study_root, args.reference_root, args.code_commit
        )))
    elif args.action == "select-stage-b":
        print(json.dumps(run_stage_b_selection(
            args.study_root, args.reference_root, args.code_commit
        )))
    elif args.action == "report-formal":
        print(json.dumps(run_formal_report(
            args.study_root, args.external_root, args.code_commit
        )))


if __name__ == "__main__":
    main()
