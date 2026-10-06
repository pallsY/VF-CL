import unittest

import torch

from stage_j_head import (
    BLEND_WEIGHT,
    CANDIDATES,
    LINEAR_REGULARIZATION,
    PROTOCOLS,
    ncm_scores,
    row_standardize,
    select,
)


class StageJHeadTests(unittest.TestCase):
    def test_four_frozen_unique_candidates(self):
        self.assertEqual(CANDIDATES, (
            "j0_tcb",
            "j1_cosine_ncm",
            "j2_l2_linear",
            "j3_tcb_ncm_blend",
        ))
        self.assertEqual(len(CANDIDATES), len(set(CANDIDATES)))
        self.assertEqual(LINEAR_REGULARIZATION, 0.01)
        self.assertEqual(BLEND_WEIGHT, 0.5)

    def test_dataset_incumbents_and_task_il_floors_are_frozen(self):
        self.assertEqual(
            PROTOCOLS["isolet"]["incumbent_name"],
            "tcb_c0.001_t0.001_g1",
        )
        self.assertEqual(PROTOCOLS["isolet"]["task_il_floor"], 0.9813)
        self.assertEqual(
            PROTOCOLS["upmc_food101"]["incumbent_name"],
            "tcb_c0.001_t0.003_g1",
        )
        self.assertEqual(PROTOCOLS["upmc_food101"]["task_il_floor"], 0.9195)

    def test_row_standardize_has_zero_mean_and_unit_variance(self):
        scores = torch.tensor([[1.0, 2.0, 4.0], [8.0, 3.0, -1.0]])
        standardized = row_standardize(scores)
        self.assertTrue(torch.allclose(
            standardized.mean(dim=1), torch.zeros(2), atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            standardized.std(dim=1), torch.ones(2), atol=1e-6
        ))

    def test_ncm_classifies_separated_centers(self):
        train_embeddings = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
        ])
        train_labels = torch.tensor([0, 0, 1, 1])
        evaluation_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scores, fit = ncm_scores(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            torch.zeros(2, 2),
            {0: [0], 1: [1]},
        )
        self.assertTrue(torch.equal(scores.argmax(1), torch.tensor([0, 1])))
        self.assertEqual(fit["centroid_count"], 2)

    def test_selection_applies_bwt_and_task_il_before_accuracy(self):
        records = [
            {"candidate": "bad_bwt", "metrics": {
                "AA_final": 0.95, "BWT": 0.1, "AA_final_taskil": 0.95,
            }},
            {"candidate": "eligible", "metrics": {
                "AA_final": 0.81, "BWT": 0.13, "AA_final_taskil": 0.93,
            }},
        ]
        result = select(
            records,
            {"er_bwt": 0.1243, "der_pp_aa_final": 0.7994},
            task_il_floor=0.9195,
        )
        self.assertEqual(result["selected_candidate"], "eligible")
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
