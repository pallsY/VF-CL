import unittest
from pathlib import Path

from frozen_ablation_smoke import (sample_order_sha256,
                                   validate_sample_orders, smoke_config)


class SmokeConfigTests(unittest.TestCase):
    def test_sample_order_digest_changes_when_epoch_order_changes(self):
        first = [{'loader_key': "('train', (0, 1))",
                  'epoch_orders': [[3, 1, 2]]}]
        second = [{'loader_key': "('train', (0, 1))",
                   'epoch_orders': [[1, 3, 2]]}]
        self.assertNotEqual(sample_order_sha256(first),
                            sample_order_sha256(second))

    def test_comparison_rejects_unrecorded_or_different_sample_orders(self):
        base = {'states': [{'task': 0, 'sample_order_sha256': 'a'},
                           {'task': 1, 'sample_order_sha256': 'b'}]}
        default = {'states': [{'task': 0, 'sample_order_sha256': 'a'},
                              {'task': 1, 'sample_order_sha256': 'b'}]}
        off = {'states': [{'task': 0, 'sample_order_sha256': 'a'},
                          {'task': 1, 'sample_order_sha256': 'c'}]}
        with self.assertRaisesRegex(ValueError, 'sample order'):
            validate_sample_orders(base, default, off)

    def test_two_task_cpu_smoke_preserves_source_and_only_changes_mask_flag(self):
        source = {'custom_tasks': '0,1|2,3|4,5', 'seed': 42,
                  'fim_freeze_frac': 0.0, 'device': 'cuda:1',
                  'epochs_per_task': 50, 'num_tasks': 3, 'num_classes': 6}
        root = Path('/tmp/smoke')
        base = smoke_config(source, 'base', root)
        on = smoke_config(source, 'default_on', root)
        off = smoke_config(source, 'off', root)
        self.assertEqual(base['num_tasks'], 3)
        self.assertEqual(base['num_classes'], 6)
        self.assertEqual(base['custom_tasks'], source['custom_tasks'])
        self.assertEqual(base['epochs_per_task'], 1)
        self.assertEqual(base['device'], 'cpu')
        self.assertEqual(base['num_workers'], 2)
        self.assertNotIn('fim_apply_mask', base)
        self.assertEqual(on['fim_apply_mask'], 1)
        self.assertEqual(off['fim_apply_mask'], 0)
        self.assertEqual(source['device'], 'cuda:1')

    def test_cifar_keeps_formal_bic_and_full_schedule(self):
        source = {'custom_tasks': '', 'classes_per_task': 10,
                  'num_tasks': 10, 'num_classes': 100,
                  'data': 'cifar100', 'bic_enabled': 1}
        config = smoke_config(source, 'base', Path('/tmp/smoke'))
        self.assertEqual(config['num_classes'], 100)
        self.assertEqual(config['num_tasks'], 10)
        self.assertEqual(config['custom_tasks'], '')
        self.assertEqual(config['bic_enabled'], 1)


if __name__ == '__main__':
    unittest.main()
