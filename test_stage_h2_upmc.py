import unittest

from stage_h2_upmc import CANDIDATES


class StageH2UPMCTests(unittest.TestCase):
    def test_candidate_budget_is_six_unique_local_tcb_variants(self):
        names = [item["name"] for item in CANDIDATES]
        self.assertEqual(len(names), 6)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(item["kind"] == "tcb" for item in CANDIDATES))

    def test_incumbent_and_only_one_axis_neighbors_are_frozen(self):
        incumbent = CANDIDATES[0]
        self.assertEqual(incumbent["name"], "tcb_c0.001_t0.001_g1")
        base = (
            incumbent["class_reg"], incumbent["task_reg"],
            incumbent["task_weight"],
        )
        for item in CANDIDATES[1:]:
            values = (
                item["class_reg"], item["task_reg"], item["task_weight"],
            )
            self.assertEqual(sum(a != b for a, b in zip(base, values)), 1)


if __name__ == "__main__":
    unittest.main()
