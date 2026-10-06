import json
import tempfile
import unittest
from pathlib import Path

from cifar100_plasticity_stage_b import (
    claim_next,
    job_specs,
    promoted_candidates,
)


class PlasticityStageBTests(unittest.TestCase):
    def _summary(self, directory):
        path = Path(directory) / "stage_a.json"
        path.write_text(json.dumps({
            "passed": True,
            "promoted_candidates": ["fim_0.00", "distill_0.25_fim_0.10"],
            "records": [],
        }), encoding="utf-8")
        return path

    def test_jobs_cover_only_remaining_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._summary(tmp)
            jobs = job_specs(summary)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"43", "44"})

    def test_claims_are_atomic_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._summary(tmp)
            claims = Path(tmp) / "claims"
            expected = job_specs(summary)
            actual = [claim_next(summary, claims) for _ in range(5)]
        self.assertEqual(actual[:4], expected)
        self.assertIsNone(actual[4])

    def test_invalid_promotion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({
                "passed": True, "promoted_candidates": ["fim_0.00"]
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                promoted_candidates(path)


if __name__ == "__main__":
    unittest.main()
