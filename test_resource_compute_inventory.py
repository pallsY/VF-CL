"""Count actual data access and configured training steps without test scores."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import shutil

import resource_compute_inventory as compute

inventory = compute.inventory


class ComputeInventoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        self.access = self.run / 'formal_access'
        self.access.mkdir()
        self.records = []
        for classes, batches in (((0, 1), 3), ((2, 3), 2)):
            for iteration in (0, 1):
                for batch in range(batches):
                    self.records.append({
                        'loader_key': repr(('train', classes)),
                        'loader_iteration': iteration, 'batch': batch,
                        'indices': [0, 1], 'shape': [2, 5],
                    })
        self.config = {'num_tasks': 2, 'epochs_per_task': 1,
                       'cl_method': 'proto_evolve',
                       'head_consolidation_enabled': 1,
                       'head_consolidation_schedule': 'final',
                       'head_full_steps': 500, 'head_bias_steps': 600,
                       'bic_enabled': 0, 'bic_steps': 200}

    def write_flow(self):
        (self.run / 'data_flow_audit.jsonl').write_text(''.join(
            json.dumps(row) + '\n' for row in self.records),
            encoding='utf-8')

    def add_access(self, split, indices=None):
        marker = {'status': 'consumed',
                  'access': {'split': split, 'phase': 'final'}}
        (self.access / f'{split}.consumed.json').write_text(
            json.dumps(marker), encoding='utf-8')
        if indices is not None:
            folder = self.run / ('validation' if split == 'validation'
                                 else 'bic')
            folder.mkdir(exist_ok=True)
            name = ('validation_manifest.json' if split == 'validation'
                    else 'calibration_manifest.json')
            (folder / name).write_text(json.dumps(
                {'ordered_indices': indices}), encoding='utf-8')

    def test_adaptive_extra_passes_and_validation_are_visible(self):
        self.write_flow()
        self.add_access('validation', [3, 7, 11])
        self.add_access('test')
        result = inventory(self.run, self.config)
        self.assertEqual(result['train_loader_passes'], 4)
        self.assertEqual(result['extra_loader_passes'], 2)
        self.assertEqual(result['train_batch_reads'], 10)
        self.assertEqual(result['current_train_loader_sample_reads'], 20)
        self.assertNotIn('train_sample_reads', result)
        self.assertEqual(result['nominal_optimizer_steps'], 5)
        self.assertEqual(result['head_candidate_steps'], 1100)
        self.assertEqual(result['validation_examples'], 3)
        self.assertEqual(result['calibration_examples'], 0)
        self.assertEqual(result['test_accesses'], 1)
        self.assertNotIn('accuracy', json.dumps(result))

    def test_baseline_configured_head_steps_are_not_counted_as_used(self):
        self.write_flow()
        self.add_access('test')
        config = dict(self.config, cl_method='er_ace',
                      head_consolidation_enabled=0)
        result = inventory(self.run, config)
        self.assertEqual(result['head_candidate_steps'], 0)
        self.assertEqual(result['validation_examples'], 0)

    def test_cifar_calibration_is_distinct_from_validation(self):
        self.write_flow()
        self.add_access('calibration', [1, 2])
        self.add_access('test')
        config = dict(self.config, cl_method='er_ace',
                      head_consolidation_enabled=0,
                      bic_enabled=1, bic_steps=1000)
        result = inventory(self.run, config)
        self.assertEqual(result['calibration_examples'], 2)
        self.assertEqual(result['validation_examples'], 0)
        self.assertEqual(result['configured_bic_steps'], 2000)

    def test_duplicate_batch_and_missing_manifest_fail_closed(self):
        self.records.append(dict(self.records[0]))
        self.write_flow()
        self.add_access('test')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inventory(self.run, self.config)
        self.records.pop()
        self.write_flow()
        self.add_access('validation')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            inventory(self.run, self.config)

    def test_pinned_access_cell_reads_no_accuracy(self):
        self.write_flow()
        self.add_access('test')
        job = self.run / 'cells/isolet/er_ace/42'
        copied = job / 'run'
        copied.mkdir(parents=True)
        shutil.copytree(self.run / 'formal_access', copied / 'formal_access')
        shutil.copy2(self.run / 'data_flow_audit.jsonl',
                     copied / 'data_flow_audit.jsonl')
        config = dict(self.config, cl_method='er_ace',
                      head_consolidation_enabled=0,
                      seed=42, num_parties=4, data_path='/data')
        (copied / 'config.json').write_text(json.dumps(config))
        (job / 'JOB_PLAN.json').write_text(json.dumps({
            'dataset': 'isolet', 'method': 'er_ace', 'seed': 42,
            'code_commit': 'a' * 40,
        }))
        authority = {'name': 'test', 'jobs_root': self.run / 'cells',
                     'methods': ('er_ace',), 'producer_commit': 'a' * 40,
                     'data_root': '/data'}
        with mock.patch.object(compute, 'verify_authority'), \
                mock.patch.dict(compute.DATASETS, {'isolet': (2, 1)}):
            row = compute.read_access_cell('isolet', 'er_ace', 42, authority)
            self.assertEqual(row['nominal_optimizer_steps'], 5)
            self.assertEqual(row['validation_examples'], 0)
            self.assertNotIn('accuracy', json.dumps(row))
            plan = json.loads((job / 'JOB_PLAN.json').read_text())
            plan['code_commit'] = 'b' * 40
            (job / 'JOB_PLAN.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'producer'):
                compute.read_access_cell('isolet', 'er_ace', 42, authority)


if __name__ == '__main__':
    unittest.main()
