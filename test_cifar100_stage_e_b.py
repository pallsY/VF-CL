import json
import tempfile
import unittest
from pathlib import Path

from cifar100_stage_e_b import job_specs, promoted_candidates


class StageEBTests(unittest.TestCase):
    def test_promoted_candidates_create_four_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({
                "passed": True,
                "promoted_candidates": [
                    "mean_l0.15_m0.00", "mean_l0.15_m0.05",
                ],
            }), encoding="utf-8")
            jobs = job_specs(path)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"43", "44"})

    def test_unknown_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({
                "passed": True,
                "promoted_candidates": ["unknown", "mean_l0.15_m0.00"],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                promoted_candidates(path)


if __name__ == "__main__": unittest.main()
