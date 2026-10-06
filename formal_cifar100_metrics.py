#!/usr/bin/env python3
"""Recompute the frozen CIFAR-100 10x10 continual-learning metrics."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


METRICS = ("aa_final_cil", "aa_avg_cil", "bwt_cil", "task_il_final")
METHOD_NAMES = {
    "finetune": "Fine-tune",
    "lwf": "LwF",
    "ewc": "EWC",
    "er": "ER",
    "der_pp": "DER++",
    "gpm": "GPM",
    "fedprotip_vfl": "FedProTIP",
}
PROTOCOL_FIELDS = (
    "data",
    "num_classes",
    "num_tasks",
    "classes_per_task",
    "num_parties",
    "model_type",
    "aggregation",
    "embed_dim",
    "epochs_per_task",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "deterministic",
)


def _mean(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric values must be non-empty and finite")
    return statistics.fmean(values)


def compute_formal_metrics(history, expected_tasks=10):
    """Compute final/stream Class-IL, final-minus-diagonal BWT, and Task-IL."""
    cil = [entry for entry in history if "CIL" in entry.get("step", "")]
    if len(cil) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} CIL entries, found {len(cil)}"
        )

    rows = []
    for task_id, entry in enumerate(cil):
        row = entry.get("per_task_accs") or {}
        expected = {f"task_{index}" for index in range(task_id + 1)}
        if set(row) != expected:
            raise ValueError(
                f"CIL entry {task_id} has tasks {sorted(row)}, "
                f"expected {sorted(expected)}"
            )
        rows.append({key: float(value) for key, value in row.items()})

    stage_accuracy = [_mean(row.values()) for row in rows]
    final = rows[-1]
    if expected_tasks == 1:
        bwt = 0.0
    else:
        bwt = _mean(
            final[f"task_{task_id}"] - rows[task_id][f"task_{task_id}"]
            for task_id in range(expected_tasks - 1)
        )

    task_il = cil[-1].get("per_task_accs_taskil") or {}
    expected_final = {f"task_{index}" for index in range(expected_tasks)}
    if set(task_il) != expected_final:
        raise ValueError("final Task-IL row is incomplete")

    return {
        "aa_final_cil": stage_accuracy[-1],
        "aa_avg_cil": _mean(stage_accuracy),
        "bwt_cil": bwt,
        "task_il_final": _mean(task_il.values()),
    }


def raw_history(results, config):
    """Normalize raw histories; FedProTIP must use its global Class-IL readout."""
    history = results.get("task_acc_history") or []
    if config.get("cl_method") != "fedprotip_vfl":
        return history

    normalized = []
    for entry in history:
        global_cil = (
            entry.get("companion_readouts", {}).get("class_il_global")
        )
        if not global_cil:
            raise ValueError("FedProTIP run is missing class_il_global")
        normalized.append(
            {
                "step": entry.get("step"),
                "per_task_accs": global_cil,
                "per_task_accs_taskil": entry.get("per_task_accs_taskil"),
            }
        )
    return normalized


def calibration_history(events):
    history = []
    for event in events:
        calibrated = event.get("paired", {}).get("calibrated", {})
        history.append(
            {
                "step": event.get("step"),
                "per_task_accs": calibrated.get("per_task_accuracy"),
                "per_task_accs_taskil": calibrated.get("task_il"),
            }
        )
    return history


def calibration_is_legal(events):
    if not events:
        return False
    for event in events:
        calibration = event.get("calibration_audit", {})
        privacy = event.get("privacy_audit", {})
        if not calibration.get("passed") or calibration.get("test_used_for_fit"):
            return False
        if not privacy.get("passed") or privacy.get("test_used_for_fit"):
            return False
        if privacy.get("raw_images_saved") or privacy.get("party_embeddings_saved"):
            return False
        if float(event.get("task_il_max_abs_delta", 0.0)) > 1e-6:
            return False
    return True


def _load_run(path):
    with path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    config = {}
    config_path = path.with_name("config.json")
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            config.update(json.load(handle))
    config.update(results.get("config") or {})
    return results, config


def _method_name(config, path):
    method = config.get("cl_method")
    name = str(config.get("exp_name", "")) + " " + str(path)
    if method == "proto_evolve" and "proto_uniform" in name:
        return "Ours (Uniform KD)"
    return METHOD_NAMES.get(method, str(method))


def _protocol(config):
    return {field: config.get(field) for field in PROTOCOL_FIELDS}


def _row(method, seed, metrics, source, legacy_bwt, calibration_legal):
    return {
        "method": method,
        "seed": int(seed),
        **metrics,
        "legacy_bwt": legacy_bwt,
        "calibration_legal": bool(calibration_legal),
        "source": str(source),
    }


def collect_external_runs(root, expected_tasks=10):
    """Collect the newest comparable result for each method/seed."""
    root = Path(root)
    candidates = []
    protocols = []
    rejected_calibrations = []
    for path in sorted(root.rglob("results.json")):
        results, config = _load_run(path)
        if str(config.get("data", "")).lower() != "cifar100":
            continue
        if int(config.get("num_tasks", 0)) != expected_tasks:
            continue

        history = raw_history(results, config)
        try:
            metrics = compute_formal_metrics(history, expected_tasks)
        except ValueError:
            continue
        method = _method_name(config, path)
        seed = config.get("seed")
        if seed is None:
            continue
        protocols.append(_protocol(config))
        candidates.append(
            (
                path.stat().st_mtime,
                _row(
                    method,
                    seed,
                    metrics,
                    path,
                    (results.get("cl_metrics") or {}).get("BWT"),
                    False,
                ),
            )
        )

        events = results.get("bic_history") or []
        if method == "Ours (Uniform KD)" and len(events) == expected_tasks:
            if calibration_is_legal(events):
                calibrated = compute_formal_metrics(
                    calibration_history(events), expected_tasks
                )
                candidates.append(
                    (
                        path.stat().st_mtime,
                        _row(
                            "Ours (Uniform KD + Joint Cal.)",
                            seed,
                            calibrated,
                            path,
                            None,
                            True,
                        ),
                    )
                )
            else:
                rejected_calibrations.append(str(path))

    unique_protocols = {
        json.dumps(protocol, sort_keys=True, ensure_ascii=False)
        for protocol in protocols
    }
    if len(unique_protocols) != 1:
        raise ValueError("external runs do not share one formal protocol")

    newest = {}
    for modified, row in candidates:
        key = (row["method"], row["seed"])
        if key not in newest or modified > newest[key][0]:
            newest[key] = (modified, row)
    rows = [item[1] for item in newest.values()]
    rows.sort(key=lambda row: (row["method"], row["seed"]))
    protocol = json.loads(next(iter(unique_protocols))) if unique_protocols else {}
    audit = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "result_files_scanned": len(list(root.rglob("results.json"))),
        "rows_selected": len(rows),
        "rejected_calibrations": rejected_calibrations,
    }
    return rows, audit


def aggregate_rows(rows, expected_seeds=(42, 43, 44)):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    summary = []
    for method, method_rows in grouped.items():
        seeds = [row["seed"] for row in method_rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {method}")
        missing = sorted(set(expected_seeds) - set(seeds))
        record = {
            "method": method,
            "n_seeds": len(seeds),
            "seeds": ",".join(map(str, sorted(seeds))),
            "missing_seeds": ",".join(map(str, missing)),
            "complete": not missing,
            "calibration_legal": all(
                row["calibration_legal"] for row in method_rows
            ),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows]
            record[f"{metric}_mean"] = _mean(values)
            record[f"{metric}_std"] = statistics.pstdev(values)
        summary.append(record)
    summary.sort(key=lambda row: row["aa_final_cil_mean"], reverse=True)
    return summary


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, rows, summary, audit):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_run_fields = (
        "method",
        "seed",
        *METRICS,
        "legacy_bwt",
        "calibration_legal",
        "source",
    )
    summary_fields = (
        "method",
        "aa_final_cil_mean",
        "aa_final_cil_std",
        "aa_avg_cil_mean",
        "aa_avg_cil_std",
        "bwt_cil_mean",
        "bwt_cil_std",
        "task_il_final_mean",
        "task_il_final_std",
        "n_seeds",
        "seeds",
        "missing_seeds",
        "complete",
        "calibration_legal",
    )
    _write_csv(output / "FORMAL_CIFAR100_PER_RUN.csv", rows, per_run_fields)
    _write_csv(output / "FORMAL_CIFAR100_TABLE.csv", summary, summary_fields)
    (output / "FORMAL_CIFAR100_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def value(row, metric):
        return "{:.2f} +/- {:.2f}".format(
            100.0 * row[f"{metric}_mean"], 100.0 * row[f"{metric}_std"]
        )

    lines = [
        "# Formal CIFAR-100 10x10 Results",
        "",
        "All values are percentages; std uses ddof=0. BWT is recomputed as "
        "final-minus-diagonal Class-IL BWT.",
        "",
        "| Method | AA-final CIL | AA-avg CIL | BWT CIL | Task-IL Acc | "
        "Seeds | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        status = "complete" if row["complete"] else (
            "missing seed " + row["missing_seeds"]
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["method"],
                value(row, "aa_final_cil"),
                value(row, "aa_avg_cil"),
                value(row, "bwt_cil"),
                value(row, "task_il_final"),
                row["seeds"],
                status,
            )
        )
    (output / "FORMAL_CIFAR100_TABLE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", default="42,43,44")
    args = parser.parse_args()
    expected_seeds = tuple(int(seed) for seed in args.expected_seeds.split(","))
    rows, audit = collect_external_runs(args.external_root)
    summary = aggregate_rows(rows, expected_seeds)
    audit["expected_seeds"] = list(expected_seeds)
    audit["missing_by_method"] = {
        row["method"]: row["missing_seeds"] for row in summary
    }
    write_outputs(args.output_dir, rows, summary, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
