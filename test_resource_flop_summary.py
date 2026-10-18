"""Task-weighted operator counts stay distinct from unmeasured head work."""

import json
import tempfile
import unittest
from pathlib import Path

from resource_flop_summary import (nominal_steps_per_task, weighted_profile,
                                   profile_name)


class FlopSummaryTests(unittest.TestCase):
    def test_cifar_profile_file_uses_frozen_short_name(self):
        self.assertEqual(profile_name('cifar100', 'sample', 0),
                         'cifar_sample_t0.json')
        self.assertEqual(profile_name('upmc', 'derpp', 9),
                         'upmc_derpp_t9.json')

    def test_weights_each_profile_by_real_task_batch_count(self):
        profiles = [
            {'profile_task': 0, 'flop_counter_total': 10,
             'probe_script_sha256': 'a' * 64},
            {'profile_task': 1, 'flop_counter_total': 20,
             'probe_script_sha256': 'a' * 64},
        ]
        result = weighted_profile(profiles, {0: 3, 1: 4})
        self.assertEqual(result['model_train_flop_proxy'], 110)
        self.assertEqual(result['nominal_optimizer_steps'], 7)
        with self.assertRaisesRegex(ValueError, 'task'):
            weighted_profile(profiles[:1], {0: 3, 1: 4})

    def test_first_loader_iteration_sets_epoch_step_count(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'flow.jsonl'
        rows = [
            {'loader_key': repr(('train', classes)),
             'loader_iteration': iteration, 'batch': batch,
             'indices': [1, 2], 'shape': [2, 5]}
            for classes, size in (((0, 1), 3), ((2, 3), 2))
            for iteration in (0, 1)
            for batch in range(size)
        ]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        config = {'custom_tasks': '0,1|2,3', 'num_tasks': 2,
                  'num_classes': 4, 'epochs_per_task': 2}
        self.assertEqual(nominal_steps_per_task(path, config),
                         {0: 6, 1: 4})


if __name__ == '__main__':
    unittest.main()
