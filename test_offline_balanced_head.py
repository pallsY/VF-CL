import json
import tempfile
import unittest
from pathlib import Path

import torch

from offline_balanced_head import (
    apply_task_class_bias,
    fit_cosine_centroids,
    fit_task_class_bias,
    replace_final_metrics,
    select_candidate,
    summarize_scores,
)


class OfflineBalancedHeadTests(unittest.TestCase):
    def test_score_summary_separates_class_and_task_accuracy(self):
        task_classes = {0: [0, 1], 1: [2, 3]}
        labels = torch.tensor([0, 1, 2, 3])
        scores = torch.tensor([
            [4.0, 1.0, 5.0, 0.0],
            [0.0, 4.0, 5.0, 1.0],
            [0.0, 1.0, 4.0, 2.0],
            [1.0, 0.0, 2.0, 4.0],
        ])
        summary = summarize_scores(scores, labels, task_classes)
        self.assertAlmostEqual(summary["overall_accuracy"], 0.5)
        self.assertAlmostEqual(summary["task_il"]["task_0"], 1.0)
        self.assertAlmostEqual(summary["task_il"]["task_1"], 1.0)

    def test_class_bias_fit_improves_systematic_class_bias(self):
        task_classes = {0: [0, 1], 1: [2, 3]}
        labels = torch.tensor([0, 1, 2, 3] * 20)
        logits = torch.zeros(len(labels), 4)
        logits[torch.arange(len(labels)), labels] = 2.0
        logits[:, 1] += 3.0
        before = summarize_scores(logits, labels, task_classes)["overall_accuracy"]
        state = fit_task_class_bias(
            logits, labels, task_classes, regularization=0.001,
            steps=200, lr=0.05,
        )
        after = summarize_scores(
            apply_task_class_bias(logits, state), labels, task_classes
        )["overall_accuracy"]
        self.assertGreater(after, before)

    def test_cosine_centroids_classify_cluster_centers(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        centroids = fit_cosine_centroids(embeddings, labels, [0, 1])
        predictions = (
            torch.nn.functional.normalize(embeddings, dim=1) @ centroids.t()
        ).argmax(1)
        self.assertTrue(torch.equal(predictions, labels))

    def test_replace_final_metrics_uses_formal_bwt_definition(self):
        history = []
        for task_id in range(10):
            per_task = {f"task_{index}": 0.8 for index in range(task_id + 1)}
            history.append({
                "step": f"event_{task_id}_CIL",
                "paired": {"calibrated": {
                    "per_task_accuracy": per_task,
                    "task_il": per_task,
                }},
            })
        final = {
            "per_task_accuracy": {f"task_{index}": 0.7 for index in range(10)},
            "task_il": {f"task_{index}": 0.75 for index in range(10)},
        }
        metrics = replace_final_metrics({"bic_history": history}, final)
        self.assertAlmostEqual(metrics["aa_final_cil"], 0.7)
        self.assertAlmostEqual(metrics["bwt_cil"], -0.1)
        self.assertAlmostEqual(metrics["task_il_final"], 0.75)

    def test_selection_uses_validation_and_bwt_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for seed in (42, 43, 44):
                path = Path(tmp) / f"{seed}.json"
                path.write_text(json.dumps({
                    "seed": seed,
                    "evaluation_source": "cifar100-train-validation",
                    "test_used_for_selection": False,
                    "records": [
                        {"candidate": "high_aa_bad_bwt", "metrics": {
                            "aa_final_cil": 0.5, "aa_avg_cil": 0.5,
                            "bwt_cil": -0.3, "task_il_final": 0.7,
                        }},
                        {"candidate": "eligible", "metrics": {
                            "aa_final_cil": 0.4, "aa_avg_cil": 0.45,
                            "bwt_cil": -0.1, "task_il_final": 0.75,
                        }},
                    ],
                }), encoding="utf-8")
                paths.append(path)
            output = Path(tmp) / "selection.json"
            selected = select_candidate(paths, output, bwt_floor=-0.2)
            self.assertEqual(selected["selected_candidate"], "eligible")
            self.assertFalse(selected["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
