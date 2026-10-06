import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class FeatureRetentionPipelineTests(unittest.TestCase):
    def test_cli_prints_stage_a_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = subprocess.check_output(
                [
                    sys.executable,
                    "feature_retention_validation.py",
                    "jobs",
                    "--stage",
                    "A",
                    "--selection-dir",
                    directory,
                ],
                text=True,
            ).splitlines()
        self.assertEqual(
            output,
            ["feat_0.05:42", "feat_0.20:42", "feat_1.00:42"],
        )

    def test_launcher_contract_is_staged_dynamic_and_power_safe(self):
        launcher = Path(
            "run_cifar100_feature_retention_validation.sh"
        ).read_text()
        for fragment in (
            "ulimit -Sn 65536",
            "memory.free,utilization.gpu",
            "MAX_GPU_UTIL=70",
            "wait_gpu",
            "select-stage-a",
            "PROMOTED_WEIGHTS",
            "select-stage-b",
            "SELECTED_FEAT_DISTILL_WEIGHT",
            "--resume_run_dir",
            "event_9_CIL.pt",
            "FEATURE_RETENTION_NO_CANDIDATE",
            "FEATURE_RETENTION_FORMAL_SUCCESS",
            "STOPPED",
        ):
            self.assertIn(fragment, launcher)
        self.assertNotIn("index % workers", launcher)

    def test_launcher_waits_for_gpu_before_claiming(self):
        launcher = Path(
            "run_cifar100_feature_retention_validation.sh"
        ).read_text()
        worker = launcher[
            launcher.index("run_worker()") : launcher.index("run_stage()")
        ]
        self.assertLess(worker.index("wait_gpu"), worker.index(" claim "))

    def test_launcher_selection_runs_only_after_workers_succeed(self):
        launcher = Path(
            "run_cifar100_feature_retention_validation.sh"
        ).read_text()
        run_stage = launcher[
            launcher.index("run_stage()") : launcher.index("select-stage-a")
        ]
        self.assertIn('if test "$status" -ne 0', run_stage)
        self.assertIn("return 1", run_stage)

    def test_smoke_runner_propagates_determinism_and_marks_failure(self):
        runner = Path("run_feature_retention_smoke_then_study.sh").read_text()
        self.assertEqual(
            runner.count("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42"),
            3,
        )
        self.assertIn('touch "$SMOKE_ROOT/SMOKE_FAILED"', runner)
        self.assertIn(
            'compare(left_ckpt["cl_state"], right_ckpt["cl_state"])', runner
        )
        self.assertNotIn("cl_method_state", runner)


if __name__ == "__main__":
    unittest.main()
