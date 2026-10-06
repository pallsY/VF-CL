import json
import tempfile
import unittest
from pathlib import Path

from cifar100_stage_d_b import job_specs, promoted_candidates


class StageDBTests(unittest.TestCase):
    def _summary(self, directory, promoted):
        path = Path(directory) / "summary.json"
        path.write_text(json.dumps({
            "passed": True,
            "promoted_candidates": promoted,
        }), encoding="utf-8")
        return path

    def test_two_new_candidates_create_four_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary(tmp, [
                "d0.25_f0.05_m0.00", "d0.10_f0.05_m0.10",
            ])
            jobs = job_specs(path)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"43", "44"})

    def test_incumbent_is_reused_not_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary(tmp, [
                "d0.25_f0.05_m0.10", "d0.10_f0.05_m0.10",
            ])
            jobs = job_specs(path)
        self.assertEqual(jobs, [
            "d0.10_f0.05_m0.10:43", "d0.10_f0.05_m0.10:44",
        ])

    def test_unknown_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary(tmp, ["unknown", "d0.10_f0.05_m0.10"])
            with self.assertRaises(ValueError):
                promoted_candidates(path)


if __name__ == "__main__":
    unittest.main()
