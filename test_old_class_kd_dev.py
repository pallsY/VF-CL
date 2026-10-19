import unittest

from run_resource_lite_pilot import expected_tasks
from run_old_class_kd_dev import build_config


class OldClassKDDevConfigTests(unittest.TestCase):
    @staticmethod
    def source(dataset):
        if dataset == 'cifar100':
            data, classes, tasks, custom, vector = (
                'cifar100', 100, 10, '', '')
        else:
            data, classes, tasks = 'tabvfl', 26, 13
            custom = expected_tasks('isolet')
            vector = '/brf/users/yangxx/VF-CL/data/isolet/isolet_vfl.npz'
        return {
            'seed': 42, 'data': data, 'num_classes': classes,
            'num_tasks': tasks, 'custom_tasks': custom,
            'classes_per_task': 10, 'vector_npz': vector,
            'num_parties': 4, 'cl_method': 'proto_evolve',
            'sample_replay_only': 1, 'resource_lite': 0,
            'formal_deferred_evaluation': True,
            'head_consolidation_enabled': 1,
            'head_consolidation_schedule': 'final',
            'head_consolidation_mode': 'adaptive_dual_branch',
            'head_consolidation_samples_per_class': 20,
            'head_consolidation_steps': 500,
            'proto_replay_gradient_enabled': 0, 'proto_sdc': False,
            'feat_distill_weight': 0.0, 'fim_freeze_frac': 0.0,
            'party_kd_enabled': 1, 'party_kd_mode': 'uniform',
            'dep_tracking_enabled': 1,
            'expected_party_kd_variant': 'uniform',
            'lambda_validation_enabled': 1,
            'bic_enabled': int(dataset == 'cifar100'),
            'bic_fit_mode': 'joint_each_stage',
            'distill_weight': 0.25,
            'epochs_per_task': 50, 'device': 'cuda:0',
        }

    def test_full_dev_uses_fresh_seed_and_locked_holdout(self):
        from run_old_class_kd_dev import SOURCE_SHA256
        config = build_config(
            self.source('isolet'), SOURCE_SHA256['isolet'],
            'isolet', 56, 'oldonly20', '/tmp/dev',
            code_commit='a' * 40,
        )
        self.assertEqual(config['num_tasks'], 13)
        self.assertEqual(config['epochs_per_task'], 50)
        self.assertEqual(config['pilot_holdout_seed'], 20261019)
        self.assertEqual(config['pilot_holdout_per_class'], 40)
        self.assertEqual(config['seed'], 56)

    def test_cifar_preflight_keeps_full_bic_task_schedule(self):
        from run_old_class_kd_dev import SOURCE_SHA256
        config = build_config(
            self.source('cifar100'), SOURCE_SHA256['cifar100'],
            'cifar100', 54, 'oldonly20', '/tmp/preflight',
            code_commit='a' * 40, smoke=True,
        )
        self.assertEqual(config['num_tasks'], 10)
        self.assertEqual(config['custom_tasks'], '')
        self.assertEqual(config['epochs_per_task'], 1)
        self.assertEqual(config['device'], 'cpu')

    def test_old_formal_seed_cannot_be_used_for_selection(self):
        from run_old_class_kd_dev import SOURCE_SHA256
        with self.assertRaisesRegex(ValueError, 'seed'):
            build_config(
                self.source('isolet'), SOURCE_SHA256['isolet'],
                'isolet', 42, 'oldonly20', '/tmp/dev',
                code_commit='a' * 40,
            )


if __name__ == '__main__':
    unittest.main()
