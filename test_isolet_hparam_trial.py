import tempfile
import unittest
from pathlib import Path

import torch

from launch_isolet_hparam_trial import (
    TrainingOnlyStop, derive_config, rank_candidate, readout_metrics,
    stop_before_deferred_evaluation,
)


class ISOLETHparamTrialTests(unittest.TestCase):
    def test_only_registered_config_values_change(self):
        source = {
            'data': 'tabvfl', 'num_classes': 26, 'num_tasks': 13,
            'classes_per_task': 2, 'num_parties': 4,
            'aggregation': 'concat', 'model_type': 'mlp',
            'seed': 42, 'device': 'cuda:0',
            'data_path': '/home/chase/Yangxx/VF-CL/data',
            'vector_npz': '/home/chase/Yangxx/VF-CL/data/isolet/isolet_vfl.npz',
            'formal_deferred_evaluation': True,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
            'head_consolidation_samples_per_class': 20,
            'lambda_validation_enabled': 1,
            'lambda_validation_per_class': 40,
            'lambda_validation_split_seed': 20260809,
            'bic_enabled': 0, 'epochs_per_task': 50,
            'batch_size': 128, 'optimizer': 'adamw',
            'unlearn_after_tasks': [13],
            'proto_lambda_a': .15,
            'distill_weight': .25,
            'feat_distill_weight': .05,
            'results_dir': '/old', 'output_dir': '/old/run',
            'exp_name': 'old',
        }
        with tempfile.TemporaryDirectory() as scratch:
            config, changed = derive_config(
                source, Path(scratch), 45, .05, .25, .05,
            )
        self.assertEqual(set(changed), {
            'seed', 'data_path', 'vector_npz',
            'formal_deferred_evaluation', 'results_dir',
            'output_dir', 'exp_name', 'proto_lambda_a',
        })
        self.assertEqual(config['proto_lambda_a'], .05)
        self.assertEqual(config['distill_weight'], .25)
        self.assertEqual(config['feat_distill_weight'], .05)
        self.assertEqual(config['head_consolidation_samples_per_class'], 20)
        self.assertEqual(config['lambda_validation_split_seed'], 20260809)

    def test_rank_prefers_cil_then_old_then_til(self):
        a = {'cil': {'accuracy': .9}, 'old_cil': {'accuracy': .8},
             'til': {'accuracy': .98}}
        b = {'cil': {'accuracy': .9}, 'old_cil': {'accuracy': .81},
             'til': {'accuracy': .97}}
        c = {'cil': {'accuracy': .91}, 'old_cil': {'accuracy': .7},
             'til': {'accuracy': .8}}
        self.assertGreater(rank_candidate(b, (.15, .25, .05)),
                           rank_candidate(a, (.15, .25, .05)))
        self.assertGreater(rank_candidate(c, (.15, .25, .05)),
                           rank_candidate(b, (.15, .25, .05)))

    def test_cil_and_task_il_readout(self):
        logits = torch.full((2, 26), -10.0)
        logits[0, 0] = 2.0
        logits[0, 24] = 3.0
        logits[1, 24] = 4.0
        result = readout_metrics(logits.log_softmax(dim=1),
                                 torch.tensor([0, 24]))
        self.assertEqual(result['cil']['correct'], 1)
        self.assertEqual(result['til']['correct'], 2)
        self.assertEqual(result['old_cil']['correct'], 0)
        self.assertEqual(result['new_cil']['correct'], 1)

    def test_stop_precedes_test_evaluation(self):
        with self.assertRaises(TrainingOnlyStop):
            stop_before_deferred_evaluation()


if __name__ == '__main__':
    unittest.main()
