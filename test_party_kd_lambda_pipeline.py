import subprocess
import unittest
from pathlib import Path

from party_kd_lambda_validation import job_specs


class PartyKDLambdaPipelineTests(unittest.TestCase):
    def test_launcher_prints_the_fifteen_job_contract(self):
        output = subprocess.check_output(
            ['bash', 'run_cifar100_party_kd_lambda_validation.sh', '--print-jobs'],
            text=True,
        ).splitlines()
        self.assertEqual(output, job_specs())

    def test_launcher_is_dynamic_power_safe_and_audit_gated(self):
        launcher = Path(
            'run_cifar100_party_kd_lambda_validation.sh'
        ).read_text()
        for fragment in (
            'ulimit -Sn 65536',
            'party_kd_lambda_validation.py" claim',
            'find-incomplete',
            '--resume_run_dir',
            '($2 + 0) >= (minimum + 0)',
            'protocol_digest.json',
            'save_task_checkpoints',
            'VALIDATION_SUCCESS',
            'STOPPED',
        ):
            self.assertIn(fragment, launcher)
        self.assertNotIn('index % workers', launcher)

    def test_selection_runs_only_after_all_workers_succeed(self):
        launcher = Path(
            'run_cifar100_party_kd_lambda_validation.sh'
        ).read_text()
        failure_gate = launcher.index('if test "$status" -ne 0')
        selection = launcher.index(
            'party_kd_lambda_validation.py" select'
        )
        self.assertLess(failure_gate, selection)
        self.assertIn('exit 1', launcher[failure_gate:selection])


if __name__ == '__main__':
    unittest.main()
