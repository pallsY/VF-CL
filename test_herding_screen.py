import tempfile
import unittest
from pathlib import Path

from analyze_cifar_herding_screen import (
    classify_herding_response, validate_herding_config,
)


class HerdingScreenTests(unittest.TestCase):
    def test_equal_memory_gate_requires_gain_and_group_guards(self):
        first = dict(examples=2500, old_examples=2250, new_examples=250,
                     correct=500, old_correct=450, new_correct=50)
        herding = dict(first, correct=600, old_correct=530, new_correct=70)
        self.assertEqual(classify_herding_response(first, herding),
                         'representative_selection_signal')
        harmed_new = dict(herding, correct=600, old_correct=560, new_correct=40)
        self.assertEqual(classify_herding_response(first, harmed_new),
                         'no_representative_selection_signal')

    def test_exact_seed47_config_and_eight_overrides(self):
        source = dict(seed=42, lambda_validation_split_seed=20260729,
                      formal_deferred_evaluation=True,
                      head_consolidation_enabled=1,
                      head_consolidation_mode='adaptive_dual_branch',
                      results_dir='/old', output_dir='/old/run',
                      exp_name='old', lr=0.001)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'root' / 'seed_47_baseline'
            config = dict(source, seed=47,
                          lambda_validation_split_seed=20261009,
                          formal_deferred_evaluation=False,
                          head_consolidation_enabled=0,
                          head_consolidation_mode='full_classifier',
                          results_dir=str(run.parent), output_dir=str(run),
                          exp_name='cifar_herding_seed47_baseline')
            changes = {key: {'source': source[key], 'pilot': config[key]}
                       for key in source if source[key] != config[key]}
            validate_herding_config(source, config, run, changes)
            with self.assertRaises(ValueError):
                validate_herding_config(source, dict(config, lr=0.002),
                                        run, changes)


if __name__ == '__main__':
    unittest.main()
