import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import adaptive_head_consolidation as adaptive
import runner
from models import TopModel


class CapacityContractTests(unittest.TestCase):
    def test_versioned_candidate_configs_preserve_v1(self):
        self.assertEqual(adaptive.adaptive_version_for_capacity(20), 1)
        self.assertEqual(adaptive.adaptive_version_for_capacity(40), 2)
        self.assertEqual(adaptive.adaptive_candidate_configs(20), {
            'full': adaptive.FULL_BRANCH_CONFIG,
            'bias': adaptive.BIAS_BRANCH_CONFIG,
        })
        self.assertEqual(adaptive.adaptive_candidate_configs(40), {
            'full': {**adaptive.FULL_BRANCH_CONFIG, 'samples_per_class': 40},
            'bias': {**adaptive.BIAS_BRANCH_CONFIG, 'samples_per_class': 40},
        })
        for bad in (0, 30, 80, True):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                adaptive.adaptive_version_for_capacity(bad)

    def test_both_v2_branches_fit_40_and_v1_default_remains_20(self):
        pre = TopModel(2, 2)
        replay = {0: torch.zeros(40, 2), 1: torch.ones(40, 2)}
        prototypes = {c: {'mean': rows.mean(0), 'std': rows.std(0).nan_to_num()}
                      for c, rows in replay.items()}
        calls = []
        def full(*args, **kwargs):
            calls.append(('full', args[5]))
            return {}
        def bias(*args, **kwargs):
            calls.append(('bias', args[8]))
            return {}
        with mock.patch.object(adaptive, 'consolidate_classifier', side_effect=full), \
                mock.patch.object(adaptive, 'consolidate_task_class_bias', side_effect=bias):
            old = adaptive.fit_adaptive_candidates(
                pre, replay, prototypes, {0: [0], 1: [1]}, 80, 1, 'cpu',
            )
            new = adaptive.fit_adaptive_candidates(
                pre, replay, prototypes, {0: [0], 1: [1]}, 80, 1, 'cpu',
                samples_per_class=40,
            )
        self.assertEqual(calls, [('full', 20), ('bias', 20),
                                 ('full', 40), ('bias', 40)])
        self.assertEqual(old.method_version, 1)
        self.assertEqual(new.method_version, 2)
        self.assertEqual(old.samples_per_class, 20)
        self.assertEqual(new.samples_per_class, 40)

    def test_checkpoint_protocol_only_adds_capacity_for_v2(self):
        args = SimpleNamespace(
            seed=42, data='tabvfl', cl_method='proto_evolve',
            num_tasks=13, num_parties=4,
            head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
            head_consolidation_samples_per_class=20,
            formal_deferred_evaluation=False,
        )
        old = runner._checkpoint_protocol(args)
        self.assertNotIn('head_consolidation_samples_per_class', old)
        args.head_consolidation_samples_per_class = 40
        new = runner._checkpoint_protocol(args)
        self.assertEqual(new['head_consolidation_samples_per_class'], 40)
        self.assertEqual({key: value for key, value in new.items()
                          if key != 'head_consolidation_samples_per_class'}, old)
    def test_resume_semantics_bind_version_to_capacity(self):
        top = TopModel(2, 2).state_dict()
        state = {
            'adaptive_method_version': 2,
            'adaptive_top_version': 0,
            'adaptive_class_order': [], 'adaptive_gate': None,
            'head_consolidation_history': [],
            'head_validation_sha256': '',
            'adaptive_pending_task_id': None,
        }
        args = SimpleNamespace(unlearn_after_tasks=[], unlearn_classes=[])
        runner._validate_adaptive_checkpoint_semantics(
            state, {'top_model': top},
            {'num_tasks': 2, 'head_consolidation_samples_per_class': 40}, args,
        )
        with self.assertRaises(ValueError):
            runner._validate_adaptive_checkpoint_semantics(
                state, {'top_model': top},
                {'num_tasks': 2, 'head_consolidation_samples_per_class': 20}, args,
            )
    def test_v2_top_state_loads_and_v1_stays_valid(self):
        for version in (1, 2):
            with self.subTest(version=version):
                top = TopModel(2, 2)
                top.set_adaptive_mixture(
                    torch.zeros(2, 2), torch.zeros(2), 0.5, [0, 1],
                    version=version,
                )
                restored = TopModel(2, 2)
                restored.load_state_dict(top.state_dict(), strict=True)
                self.assertEqual(int(restored._adaptive_version), version)
                torch.testing.assert_close(restored(torch.ones(1, 2)),
                                           top(torch.ones(1, 2)), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
