import json
import tempfile
import unittest
from pathlib import Path

from cifar100_plasticity_formal import (
    claim_next,
    expected_formal_config,
    job_specs,
    selected_candidate,
)


class PlasticityFormalTests(unittest.TestCase):
    def _selection(self, directory):
        path = Path(directory) / "selection.json"
        path.write_text(json.dumps({
            "passed": True,
            "selected_candidate": "distill_0.25_fim_0.10",
            "selection_source": "cifar100-train-validation",
            "test_used_for_selection": False,
        }), encoding="utf-8")
        return path

    def test_jobs_cover_three_formal_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self._selection(tmp)
            jobs = job_specs(selection)
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"42", "43", "44"})

    def test_formal_config_disables_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self._selection(tmp)
            config = expected_formal_config(
                "distill_0.25_fim_0.10:42", selection
            )
        self.assertEqual(config["lambda_validation_enabled"], 0)
        self.assertEqual(config["distill_weight"], 0.25)
        self.assertEqual(config["fim_freeze_frac"], 0.10)
        self.assertEqual(config["feat_distill_weight"], 0.05)

    def test_claims_are_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self._selection(tmp)
            claims = Path(tmp) / "claims"
            actual = [claim_next(selection, claims) for _ in range(4)]
            expected = job_specs(selection)
        self.assertEqual(actual[:3], expected)
        self.assertIsNone(actual[3])

    def test_illegal_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({
                "passed": True,
                "selected_candidate": "distill_0.25_fim_0.10",
                "selection_source": "test",
                "test_used_for_selection": True,
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                selected_candidate(path)


if __name__ == "__main__":
    unittest.main()
