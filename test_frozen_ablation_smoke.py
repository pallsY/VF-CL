import unittest
from pathlib import Path

from frozen_ablation_smoke import smoke_config


class SmokeConfigTests(unittest.TestCase):
    def test_two_task_cpu_smoke_preserves_source_and_only_changes_mask_flag(self):
        source = {'custom_tasks': '0,1|2,3|4,5', 'seed': 42,
                  'fim_freeze_frac': 0.0, 'device': 'cuda:1',
                  'epochs_per_task': 50}
        root = Path('/tmp/smoke')
        base = smoke_config(source, 'base', root)
        on = smoke_config(source, 'default_on', root)
        off = smoke_config(source, 'off', root)
        self.assertEqual(base['num_tasks'], 2)
        self.assertEqual(base['num_classes'], 4)
        self.assertEqual(base['custom_tasks'], '0,1|2,3')
        self.assertEqual(base['epochs_per_task'], 1)
        self.assertEqual(base['device'], 'cpu')
        self.assertEqual(base['num_workers'], 2)
        self.assertNotIn('fim_apply_mask', base)
        self.assertEqual(on['fim_apply_mask'], 1)
        self.assertEqual(off['fim_apply_mask'], 0)
        self.assertEqual(source['device'], 'cuda:1')


if __name__ == '__main__':
    unittest.main()
