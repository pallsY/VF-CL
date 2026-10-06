import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feature_retention_validation import (
    WEIGHTS,
    audit_candidate_run,
    audit_formal_run,
    build_formal_rows,
    claim_next_job,
    command_for,
    expected_formal_config,
    expected_validation_config,
    load_reference_record,
    parse_job,
    select_stage_a,
    select_stage_b,
    stage_job_specs,
    validation_job_specs,
    write_formal_comparison,
    write_stage_a_selection,
    write_stage_b_selection,
)
from party_kd_lambda_validation import expected_config as lambda_expected_config


def make_history(final_value):
    history = []
    for stage in range(10):
        history.append({
            "step": f"event_{stage}_CIL",
            "per_task_accs": {
                f"task_{task}": final_value if stage == 9 else 0.8 - 0.01 * task
                for task in range(stage + 1)
            },
            "per_task_accs_taskil": {
                f"task_{task}": 0.7 for task in range(stage + 1)
            },
        })
    return history


def make_bic_history(final_value):
    return [{
        "step": entry["step"],
        "paired": {"calibrated": {
            "per_task_accuracy": entry["per_task_accs"],
            "task_il": entry["per_task_accs_taskil"],
        }},
        "calibration_audit": {"passed": True, "test_used_for_fit": False},
        "privacy_audit": {
            "passed": True,
            "test_used_for_fit": False,
            "raw_images_saved": False,
            "party_embeddings_saved": False,
        },
        "task_il_max_abs_delta": 0.0,
    } for entry in make_history(final_value)]


def metric_record(weight, seed, aa, task_il, bwt):
    return {
        "job": f"feat_{weight}:{seed}",
        "weight": weight,
        "seed": seed,
        "metrics": {
            "aa_final_cil": aa,
            "aa_avg_cil": aa,
            "task_il_final": task_il,
            "bwt_cil": bwt,
        },
        "calibration_manifest_sha256": "a" * 64,
        "validation_manifest_sha256": "b" * 64,
    }


class FeatureRetentionMatrixTests(unittest.TestCase):
    def test_stage_a_has_exactly_three_seed42_jobs(self):
        self.assertEqual(WEIGHTS, ("0.05", "0.20", "1.00"))
        self.assertEqual(
            validation_job_specs(WEIGHTS, (42,)),
            ["feat_0.05:42", "feat_0.20:42", "feat_1.00:42"],
        )

    def test_validation_config_changes_only_feature_weight_from_lambda_one(self):
        actual = expected_validation_config("feat_0.20:42")
        self.assertEqual(actual["party_kd_lambda"], 1.0)
        self.assertEqual(actual["feat_distill_weight"], 0.20)
        self.assertEqual(actual["lambda_validation_enabled"], 1)
        self.assertEqual(actual["bic_split_seed"], 20260722)
        self.assertEqual(actual["lambda_validation_split_seed"], 20260729)
        self.assertEqual(actual["save_task_checkpoints"], 3)

    def test_formal_config_closes_only_extra_validation_holdout(self):
        validation = expected_validation_config("feat_0.05:43")
        formal = expected_formal_config("0.05", 43)
        self.assertEqual(formal["lambda_validation_enabled"], 0)
        self.assertEqual(formal["feat_distill_weight"], 0.05)
        for key in validation:
            if key != "lambda_validation_enabled":
                self.assertEqual(formal[key], validation[key])

    def test_parse_job_rejects_unknown_weight_or_seed(self):
        self.assertEqual(parse_job("feat_0.20:44"), ("0.20", 44))
        for spec in ("feat_0.50:42", "feat_0.20:45", "lambda_1.00:42"):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, "unknown job"):
                    parse_job(spec)

    def test_command_names_unique_run_and_preserves_validation_mode(self):
        command = command_for(
            "feat_1.00:42", "validation", "/study", "/repo", "/python"
        )
        pairs = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(command[:2], ["/python", "/repo/main.py"])
        self.assertEqual(pairs["--feat_distill_weight"], "1.0")
        self.assertEqual(pairs["--lambda_validation_enabled"], "1")
        self.assertEqual(pairs["--results_dir"], "/study/validation/runs")
        self.assertEqual(
            pairs["--exp_name"],
            "cifar100_feature_retention_validation_feat_1p00_seed42",
        )

    def test_stage_jobs_follow_saved_promotion_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory)
            self.assertEqual(
                stage_job_specs("A", selection),
                ["feat_0.05:42", "feat_0.20:42", "feat_1.00:42"],
            )
            (selection / "PROMOTED_WEIGHTS").write_text("0.20\n0.05\n")
            self.assertEqual(
                stage_job_specs("B", selection),
                [
                    "feat_0.20:43", "feat_0.20:44",
                    "feat_0.05:43", "feat_0.05:44",
                ],
            )
            (selection / "SELECTED_FEAT_DISTILL_WEIGHT").write_text("0.20\n")
            self.assertEqual(
                stage_job_specs("formal", selection),
                ["feat_0.20:42", "feat_0.20:43", "feat_0.20:44"],
            )

    def test_claims_each_stage_job_once(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = ["feat_0.05:42", "feat_0.20:42"]
            self.assertEqual(claim_next_job(jobs, directory), jobs[0])
            self.assertEqual(claim_next_job(jobs, directory), jobs[1])
            self.assertIsNone(claim_next_job(jobs, directory))


class FeatureRetentionSelectionTests(unittest.TestCase):
    def test_stage_a_requires_all_three_gates_and_promotes_at_most_two(self):
        baseline = metric_record("0.00", 42, 0.249, 0.714, -0.28)
        lwf = metric_record("lwf", 42, 0.243, 0.750, -0.43)
        candidates = [
            metric_record("0.05", 42, 0.251, 0.715, -0.27),
            metric_record("0.20", 42, 0.252, 0.716, -0.26),
            metric_record("1.00", 42, 0.260, 0.700, -0.25),
        ]
        self.assertEqual(
            select_stage_a(candidates, baseline, lwf), ["0.20", "0.05"]
        )

    def test_stage_b_returns_none_if_taskil_drops_even_when_aa_improves(self):
        candidate = [
            metric_record("0.20", seed, 0.260, 0.700, -0.25)
            for seed in (42, 43, 44)
        ]
        baseline = [
            metric_record("0.00", seed, 0.249, 0.714, -0.28)
            for seed in (42, 43, 44)
        ]
        lwf = [
            metric_record("lwf", seed, 0.243, 0.750, -0.43)
            for seed in (42, 43, 44)
        ]
        self.assertIsNone(select_stage_b(candidate, baseline, lwf))

    def test_stage_b_uses_aa_then_taskil_then_bwt_and_lower_weight_tiebreak(self):
        candidates = []
        for seed in (42, 43, 44):
            candidates.extend((
                metric_record("0.05", seed, 0.260, 0.720, -0.25),
                metric_record("0.20", seed, 0.260, 0.721, -0.26),
            ))
        baseline = [
            metric_record("0.00", seed, 0.249, 0.714, -0.28)
            for seed in (42, 43, 44)
        ]
        lwf = [
            metric_record("lwf", seed, 0.243, 0.750, -0.43)
            for seed in (42, 43, 44)
        ]
        self.assertEqual(select_stage_b(candidates, baseline, lwf)["weight"], "0.20")

    def test_selection_rejects_manifest_mismatch(self):
        baseline = metric_record("0.00", 42, 0.249, 0.714, -0.28)
        lwf = metric_record("lwf", 42, 0.243, 0.750, -0.43)
        candidate = metric_record("0.05", 42, 0.251, 0.715, -0.27)
        candidate["validation_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identical manifests"):
            select_stage_a([candidate], baseline, lwf)

    def test_candidate_audit_rejects_test_selection_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = expected_validation_config("feat_0.05:42")
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "results.json").write_text(json.dumps({
                "task_acc_history": make_history(0.2),
                "bic_history": make_bic_history(0.4),
                "calibration_audit": {
                    "passed": True, "test_used_for_fit": False,
                },
                "selection_audit": {
                    "passed": False,
                    "test_used_for_selection": True,
                    "evaluation_source": "cifar100-test",
                    "training_calibration_overlap_count": 0,
                    "training_validation_overlap_count": 0,
                    "calibration_validation_overlap_count": 0,
                    "calibration_manifest_sha256": "a" * 64,
                    "validation_manifest_sha256": "b" * 64,
                },
            }))
            checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "selection audit failed"):
                audit_candidate_run(
                    "feat_0.05:42", run_dir, "deadbeef", "validation"
                )

    def test_reference_loader_recomputes_without_rewriting_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "cifar100_lambda_validation_lambda_1p00_seed42_test"
            run_dir.mkdir(parents=True)
            config = lambda_expected_config("lambda_1.00:42")
            result = {
                "task_acc_history": make_history(0.2),
                "bic_history": make_bic_history(0.4),
            }
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "results.json").write_text(json.dumps(result))
            checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            from party_kd_lambda_validation import metrics_for_result
            digest = {
                "job": "lambda_1.00:42",
                "config": config,
                "metrics": metrics_for_result("lambda_1.00:42", result),
                "calibration_manifest_sha256": "a" * 64,
                "validation_manifest_sha256": "b" * 64,
                "code_commit": "8d63af5",
            }
            digest_path = run_dir / "protocol_digest.json"
            digest_path.write_text(json.dumps(digest))
            before = digest_path.read_bytes()

            record = load_reference_record("lambda_1.00:42", root)

            self.assertEqual(record["weight"], "0.00")
            self.assertEqual(record["seed"], 42)
            self.assertEqual(digest_path.read_bytes(), before)

    def test_stage_a_no_candidate_writes_terminal_marker_only(self):
        baseline = metric_record("0.00", 42, 0.249, 0.714, -0.28)
        lwf = metric_record("lwf", 42, 0.243, 0.750, -0.43)
        candidate = metric_record("0.05", 42, 0.240, 0.715, -0.27)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            promoted = write_stage_a_selection([candidate], baseline, lwf, root)
            self.assertEqual(promoted, [])
            self.assertTrue((root / "FEATURE_RETENTION_NO_CANDIDATE").is_file())
            self.assertFalse((root / "selection" / "PROMOTED_WEIGHTS").exists())

    def test_stage_b_writes_fixed_weight_without_terminal_marker(self):
        candidates = [
            metric_record("0.20", seed, 0.260, 0.720, -0.25)
            for seed in (42, 43, 44)
        ]
        baseline = [
            metric_record("0.00", seed, 0.249, 0.714, -0.28)
            for seed in (42, 43, 44)
        ]
        lwf = [
            metric_record("lwf", seed, 0.243, 0.750, -0.43)
            for seed in (42, 43, 44)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = write_stage_b_selection(candidates, baseline, lwf, root)
            self.assertEqual(selected["weight"], "0.20")
            self.assertEqual(
                (root / "selection" / "SELECTED_FEAT_DISTILL_WEIGHT").read_text(),
                "0.20\n",
            )
            self.assertFalse((root / "FEATURE_RETENTION_NO_CANDIDATE").exists())


def formal_reference_rows():
    rows = []
    for method in (
        "LwF",
        "Ours (Uniform KD)",
        "Ours (Uniform KD + Joint Cal.)",
    ):
        for seed in (42, 43, 44):
            rows.append({
                "method": method,
                "seed": seed,
                "aa_final_cil": 0.2,
                "aa_avg_cil": 0.3,
                "bwt_cil": -0.4,
                "task_il_final": 0.7,
                "legacy_bwt": None,
                "calibration_legal": "Joint Cal." in method,
                "source": f"/{method}/{seed}",
            })
    return rows


def formal_candidate_records(weight="0.20"):
    return [{
        "weight": weight,
        "seed": seed,
        "metrics": {
            "aa_final_cil": 0.26,
            "aa_avg_cil": 0.40,
            "bwt_cil": -0.25,
            "task_il_final": 0.72,
        },
        "result": f"/candidate/{seed}/results.json",
        "run_dir": f"/candidate/{seed}",
    } for seed in (42, 43, 44)]


class FeatureRetentionFormalTests(unittest.TestCase):
    def test_formal_rows_contain_only_reference_and_fixed_candidate(self):
        rows = build_formal_rows(
            formal_reference_rows(), formal_candidate_records(), "0.20"
        )
        self.assertEqual(
            {row["method"] for row in rows},
            {
                "LwF",
                "Ours (Uniform KD)",
                "Ours (Uniform KD + Joint Cal.)",
                "Ours (Uniform KD + feat-KD=0.20 + Joint Cal.)",
            },
        )

    def test_formal_audit_rejects_validation_holdout_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = expected_formal_config("0.20", 42)
            config["lambda_validation_enabled"] = 1
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "results.json").write_text(json.dumps({
                "bic_history": make_bic_history(0.4),
            }))
            checkpoint = run_dir / "checkpoints" / "event_9_CIL.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "formal protocol mismatch"):
                audit_formal_run("feat_0.20:42", run_dir, "deadbeef")

    def test_formal_report_renames_outputs_and_records_fixed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "feature_retention_validation.collect_external_runs",
                return_value=(formal_reference_rows(), {"passed": True}),
            ):
                summary = write_formal_comparison(
                    "/external", formal_candidate_records(), "0.20", output
                )
            self.assertEqual(len(summary), 4)
            for name in (
                "FORMAL_FEATURE_RETENTION_PER_RUN.csv",
                "FORMAL_FEATURE_RETENTION_TABLE.csv",
                "FORMAL_FEATURE_RETENTION_AUDIT.json",
                "FORMAL_FEATURE_RETENTION_TABLE.md",
            ):
                self.assertTrue((output / name).is_file(), name)
            audit = json.loads(
                (output / "FORMAL_FEATURE_RETENTION_AUDIT.json").read_text()
            )
            self.assertEqual(audit["selected_feat_distill_weight"], 0.20)
            self.assertFalse(audit["selection_test_used"])


if __name__ == "__main__":
    unittest.main()
