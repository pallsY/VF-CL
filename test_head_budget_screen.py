import unittest
import tempfile
from pathlib import Path

from analyze_cifar_head_budget import (
    classify_budget_response, validate_budget_config,
)


class HeadBudgetDecisionTests(unittest.TestCase):
    def test_only_eight_registered_config_differences_are_accepted(self):
        source = dict(seed=42, lambda_validation_split_seed=20260729,
                      formal_deferred_evaluation=True,
                      head_consolidation_enabled=1,
                      head_consolidation_mode='adaptive_dual_branch',
                      results_dir='/old', output_dir='/old/run',
                      exp_name='old', lr=0.001)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'root' / 'seed_46_baseline'
            config = dict(source, seed=46,
                          lambda_validation_split_seed=20261008,
                          formal_deferred_evaluation=False,
                          head_consolidation_enabled=0,
                          head_consolidation_mode='full_classifier',
                          results_dir=str(run.parent), output_dir=str(run),
                          exp_name='cifar_head_budget_seed46_baseline')
            changes = {key: {'source': source[key], 'pilot': config[key]}
                       for key in source if source[key] != config[key]}
            validate_budget_config(source, config, run, changes)
            with self.assertRaises(ValueError):
                validate_budget_config(source, dict(config, lr=0.002),
                                       run, changes)

    def test_prespecified_three_way_decision(self):
        small = dict(examples=2500, old_examples=2250, new_examples=250,
                     correct=500, old_correct=450, new_correct=50, nll=3.0)
        sufficient = dict(small, correct=650, old_correct=580,
                          new_correct=70, nll=2.8)
        self.assertEqual(classify_budget_response(small, sufficient),
                         'sample_sufficiency_signal')
        saturated = dict(small, correct=520, old_correct=470,
                         new_correct=50, nll=3.0)
        self.assertEqual(classify_budget_response(small, saturated),
                         'early_saturation')
        uncertain = dict(small, correct=550, old_correct=500,
                         new_correct=50, nll=2.9)
        self.assertEqual(classify_budget_response(small, uncertain),
                         'inconclusive')


if __name__ == '__main__':
    unittest.main()
