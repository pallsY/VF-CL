import json
import tempfile
import unittest
from pathlib import Path

from cifar100_stage_e_formal import expected_formal_config, job_specs


class StageEFormalTests(unittest.TestCase):
    def _selection(self, directory):
        path = Path(directory) / "selection.json"
        path.write_text(json.dumps({
            "passed": True,
            "selected_candidate": "mean_l0.15_m0.00",
            "selection_source": "cifar100-train-validation",
            "test_used_for_selection": False,
        }), encoding="utf-8")
        return path

    def test_formal_jobs_cover_three_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = job_specs(self._selection(tmp))
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job.rsplit(":", 1)[1] for job in jobs}, {"42", "43", "44"})

    def test_formal_config_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self._selection(tmp)
            config = expected_formal_config("mean_l0.15_m0.00:42", selection)
        self.assertEqual(config["lambda_validation_enabled"], 0)
        self.assertEqual(config["proto_replay_loss_norm"], "sample_mean")
        self.assertEqual(config["proto_lambda_a"], 0.15)
        self.assertEqual(config["fim_freeze_frac"], 0.0)
        self.assertEqual(config["distill_weight"], 0.25)

    def test_test_based_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._selection(tmp)
            data = json.loads(path.read_text())
            data["test_used_for_selection"] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                job_specs(path)


if __name__ == "__main__": unittest.main()
