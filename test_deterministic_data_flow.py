import random
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import VFLDataset, make_deterministic_loader
from determinism import tensor_sha256
from vfl_trainer import VFLTrainer


class RandomizedToyDataset(Dataset):
    def __len__(self):
        return 16

    targets = [index % 2 for index in range(16)]

    def __getitem__(self, index):
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return torch.tensor([float(index), noise], dtype=torch.float32), index % 2


class FormalImageDataset(Dataset):
    targets = [index % 4 for index in range(16)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index)), self.targets[index]


def collect_two_epochs(loader):
    batches = []
    for _ in range(2):
        batches.append([
            (tensor_sha256(x), y.tolist())
            for x, y in loader
        ])
    return batches, loader.audit_sampler.epoch_orders


class DeterministicDataFlowTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(seed=42, batch_size=4, num_workers=2)
        self.indices = list(range(16))
        self.key = ('train', tuple(range(10)))

    def test_same_seed_key_replays_sampler_and_worker_augmentation_across_epochs(self):
        first = make_deterministic_loader(
            RandomizedToyDataset(), self.indices, self.args, self.key, shuffle=True, audit=True
        )
        first_batches, first_orders = collect_two_epochs(first)

        random.random()
        np.random.rand()
        torch.rand(10)

        second = make_deterministic_loader(
            RandomizedToyDataset(), self.indices, self.args, self.key, shuffle=True, audit=True
        )
        second_batches, second_orders = collect_two_epochs(second)

        self.assertEqual(first_orders, second_orders)
        self.assertEqual(first_batches, second_batches)

    def test_loader_uses_two_nonpersistent_workers_and_separate_generators(self):
        loader = make_deterministic_loader(
            RandomizedToyDataset(), self.indices, self.args, self.key, shuffle=True, audit=True
        )
        self.assertEqual(loader.num_workers, 2)
        self.assertFalse(loader.persistent_workers)
        self.assertIsNot(loader.audit_sampler.generator, loader.generator)

    def test_different_task_keys_produce_different_sampler_orders(self):
        first = make_deterministic_loader(
            RandomizedToyDataset(), self.indices, self.args, ('train', (0, 1)), shuffle=True, audit=True
        )
        second = make_deterministic_loader(
            RandomizedToyDataset(), self.indices, self.args, ('train', (2, 3)), shuffle=True, audit=True
        )
        list(first)
        list(second)
        self.assertNotEqual(first.audit_sampler.epoch_orders[0], second.audit_sampler.epoch_orders[0])

    def test_trainer_audit_records_original_indices_and_cpu_post_augmentation_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = object.__new__(VFLTrainer)
            trainer.args = SimpleNamespace(data_flow_audit=1, output_dir=tmp)
            loader = SimpleNamespace(
                audit_sampler=SimpleNamespace(epoch_orders=[[10, 11, 12, 13]]),
                audit_key="('train', (0, 1))",
                batch_size=2,
            )
            batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

            trainer._record_data_flow(loader, 0, batch)

            path = os.path.join(tmp, 'data_flow_audit.jsonl')
            with open(path, encoding='utf-8') as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record['indices'], [10, 11])
            self.assertEqual(record['batch_sha256'], tensor_sha256(batch))
            self.assertEqual(record['dtype'], 'torch.float32')
            self.assertEqual(record['shape'], [2, 2])
            self.assertEqual(record['loader_iteration'], 0)

    def test_adaptive_validation_first_iteration_is_audited_when_optional_audit_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VFLDataset.__new__(VFLDataset)
            dataset.args = SimpleNamespace(
                lambda_validation_enabled=1,
                deterministic=0,
                data_flow_audit=0,
                head_consolidation_enabled=1,
                head_consolidation_mode='adaptive_dual_branch',
                output_dir=tmp,
                batch_size=2,
                num_workers=0,
            )
            dataset.validationset = RandomizedToyDataset()
            dataset.validation_indices = set(range(16))
            loader = dataset.get_validation_loader([0, 1])
            path = os.path.join(tmp, 'data_flow_audit.jsonl')

            self.assertFalse(os.path.exists(path))
            list(loader)
            list(loader)
            self.assertTrue(os.path.exists(path), 'validation first access was not audited')

            with open(path, encoding='utf-8') as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 1)
            self.assertEqual(records, [{
                'event': 'first_iteration',
                'loader_key': "('lambda_validation', (0, 1))",
                'split': 'validation',
            }])

    def test_adaptive_test_first_iteration_is_audited_when_optional_audit_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VFLDataset.__new__(VFLDataset)
            dataset.args = SimpleNamespace(
                deterministic=0,
                data_flow_audit=0,
                head_consolidation_enabled=1,
                head_consolidation_mode='adaptive_dual_branch',
                output_dir=tmp,
                batch_size=2,
                num_workers=0,
            )
            dataset.testset = RandomizedToyDataset()
            self.assertTrue(hasattr(dataset, 'get_test_loader'))
            loader = dataset.get_test_loader([0, 1])
            path = os.path.join(tmp, 'data_flow_audit.jsonl')
            self.assertFalse(os.path.exists(path))
            list(loader)
            self.assertTrue(os.path.exists(path), 'test first access was not audited')
            with open(path, encoding='utf-8') as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record['event'], 'first_iteration')
            self.assertEqual(record['split'], 'test')
            self.assertEqual(record['loader_key'], "('test', (0, 1))")

    def test_legacy_validation_keeps_first_access_audit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VFLDataset.__new__(VFLDataset)
            dataset.args = SimpleNamespace(
                lambda_validation_enabled=1,
                deterministic=0,
                data_flow_audit=0,
                output_dir=tmp,
                batch_size=2,
                num_workers=0,
                head_consolidation_enabled=1,
                head_consolidation_mode='full_classifier',
            )
            dataset.validationset = RandomizedToyDataset()
            dataset.validation_indices = set(range(16))
            list(dataset.get_validation_loader([0, 1]))
            self.assertFalse(os.path.exists(os.path.join(tmp, 'data_flow_audit.jsonl')))

    def _formal_dataset(self, output_dir):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.args = SimpleNamespace(
            formal_deferred_evaluation=True,
            num_tasks=2, classes_per_task=2, custom_tasks='',
            lambda_validation_enabled=1, deterministic=0, data_flow_audit=0,
            bic_enabled=0,
            head_consolidation_enabled=0, head_consolidation_mode='full_classifier',
            output_dir=output_dir, batch_size=2, num_workers=0,
        )
        dataset.trainset = FormalImageDataset()
        dataset.validationset = FormalImageDataset()
        dataset.testset = FormalImageDataset()
        dataset.calibrationset = FormalImageDataset()
        dataset.calibration_indices = set()
        dataset.validation_indices = set(range(16))
        return dataset

    def test_formal_calibration_requires_exact_durable_post_freeze_access(self):
        full = [0, 1, 2, 3]
        metadata = {
            'phase': 'final_bic_calibration_post_freeze',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.args.bic_enabled = 1
            dataset.calibration_indices = set(range(16))
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                dataset.get_calibration_loader(full)
            authorize = getattr(
                dataset, 'authorize_formal_calibration_access', None
            )
            self.assertTrue(callable(authorize))
            with self.assertRaisesRegex(ValueError, 'phase'):
                authorize(**{**metadata, 'phase': 'final_test_post_install'})
            authorize(**metadata)
            loader = dataset.get_calibration_loader(full)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                dataset.get_calibration_loader(full)
            list(loader)
            dataset.authorize_formal_access(
                split='test', phase='final_test_post_install',
                event_idx=1, task_id=1, timeline_step='event_1_CIL',
                classes=full,
            )
            list(dataset.get_test_loader(full))
            records = [
                json.loads(line) for line in
                (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(
                [(record['split'], record['phase']) for record in records],
                [
                    ('calibration', 'final_bic_calibration_post_freeze'),
                    ('test', 'final_test_post_install'),
                ],
            )
            self.assertEqual(
                records[0]['loader_key'],
                "('bic_calibration', (0, 1, 2, 3))",
            )
            restarted = self._formal_dataset(tmp)
            restarted.args.bic_enabled = 1
            with self.assertRaisesRegex(ValueError, 'duplicate|stale|consumed'):
                restarted.authorize_formal_calibration_access(**metadata)

    def test_formal_authorization_is_consumed_once_with_exact_phase_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.args.head_consolidation_enabled = 1
            metadata = {
                'phase': 'final_validation_pre_install',
                'event_idx': 1, 'task_id': 1,
                'timeline_step': 'event_1_CIL', 'classes': [0, 1, 2, 3],
            }
            dataset.authorize_formal_access(split='validation', **metadata)
            loader = dataset.get_validation_loader([0, 1, 2, 3])
            path = os.path.join(tmp, 'data_flow_audit.jsonl')
            self.assertFalse(os.path.exists(path))
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                dataset.get_validation_loader([0, 1, 2, 3])

            list(loader)
            with self.assertRaisesRegex(RuntimeError, 'once|iterat|consum'):
                list(loader)

            with open(path, encoding='utf-8') as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records, [{
                'event': 'first_iteration',
                'loader_key': "('lambda_validation', (0, 1, 2, 3))",
                'split': 'validation',
                'phase': 'final_validation_pre_install',
                'event_idx': 1,
                'task_id': 1,
                'timeline_step': 'event_1_CIL',
                'classes': [0, 1, 2, 3],
            }])
            with self.assertRaisesRegex(ValueError, 'duplicate|stale'):
                dataset.authorize_formal_access(split='validation', **metadata)

    def test_formal_authorization_rejects_missing_early_wrong_and_unconsumed_access(self):
        full = [0, 1, 2, 3]
        valid = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                dataset.get_test_loader(full)

        invalid = [
            ({**valid, 'event_idx': 0, 'task_id': 0,
              'timeline_step': 'event_0_CIL'}, 'early|stale'),
            ({**valid, 'phase': 'final_validation_pre_install'}, 'phase'),
            ({**valid, 'timeline_step': 'event_0_CIL'}, 'timeline'),
            ({**valid, 'classes': [0, 1, 2]}, 'classes'),
            ({**valid, 'classes': [0, 1, 2, 3, 4]}, 'classes'),
            ({**valid, 'classes': [0, 1, 2, True]}, 'classes'),
            ({**valid, 'classes': (0, 1, 2, 3)}, 'classes'),
            ({**valid, 'event_idx': True}, 'event_idx'),
        ]
        for authorization, message in invalid:
            with self.subTest(authorization=authorization), tempfile.TemporaryDirectory() as tmp:
                dataset = self._formal_dataset(tmp)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    dataset.authorize_formal_access(**authorization)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.validation_indices = {0, 1, 2, 3}
            dataset.authorize_formal_access(**valid)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                dataset.authorize_formal_access(**valid)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                dataset.get_train_loader([0, 1])
            with self.assertRaisesRegex(ValueError, 'split'):
                dataset.get_validation_loader(full)
            with self.assertRaisesRegex(ValueError, 'classes|key'):
                dataset.get_test_loader([3, 2, 1, 0])

    def test_formal_access_is_durable_across_instances_and_restart(self):
        full = [0, 1, 2, 3]
        access = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = self._formal_dataset(tmp)
            first.authorize_formal_access(**access)
            marker_dir = Path(tmp) / 'formal_access'
            pending = marker_dir / 'FORMAL_ACCESS_PENDING.json'
            self.assertTrue(pending.is_file())
            list(first.get_test_loader(full))
            self.assertFalse(pending.exists())
            self.assertTrue((marker_dir / 'test.consumed.json').is_file())

            restarted = self._formal_dataset(tmp)
            with self.assertRaisesRegex(ValueError, 'duplicate|stale|consumed'):
                restarted.authorize_formal_access(**access)
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['classes'], full)

        with tempfile.TemporaryDirectory() as tmp:
            first = self._formal_dataset(tmp)
            first.authorize_formal_access(**access)
            restarted = self._formal_dataset(tmp)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                restarted.authorize_formal_access(**access)
            self.assertFalse((Path(tmp) / 'data_flow_audit.jsonl').exists())

    def test_all_formal_loader_wrappers_are_exactly_one_shot(self):
        full = [0, 1, 2, 3]

        class CountingLoader:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(((torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),))

            def __len__(self):
                return 1

        cases = (
            ('validation', 'final_validation_pre_install',
             'get_validation_loader'),
            ('calibration', 'final_bic_calibration_post_freeze',
             'get_calibration_loader'),
            ('test', 'final_test_post_install', 'get_test_loader'),
        )
        for split, phase, getter_name in cases:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                dataset = self._formal_dataset(tmp)
                dataset.args.head_consolidation_enabled = int(split == 'validation')
                dataset.args.bic_enabled = int(split == 'calibration')
                if split == 'calibration':
                    dataset.calibration_indices = set(range(16))
                dataset.authorize_formal_access(
                    split=split, phase=phase,
                    event_idx=1, task_id=1,
                    timeline_step='event_1_CIL', classes=full,
                )
                underlying = CountingLoader()
                patcher = (
                    mock.patch('data_utils.DataLoader', return_value=underlying)
                    if split == 'calibration'
                    else mock.patch.object(
                        dataset, '_loader', return_value=underlying
                    )
                )
                with patcher:
                    loader = getattr(dataset, getter_name)(full)

                list(loader)
                with self.assertRaisesRegex(RuntimeError, 'once|iterat|consum'):
                    list(loader)
                self.assertEqual(underlying.iterations, 1)
                records = [
                    json.loads(line)
                    for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
                ]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['split'], split)

                restarted = self._formal_dataset(tmp)
                restarted.args.head_consolidation_enabled = int(
                    split == 'validation'
                )
                restarted.args.bic_enabled = int(split == 'calibration')
                if split == 'calibration':
                    restarted.calibration_indices = set(range(16))
                with self.assertRaisesRegex(
                        (RuntimeError, ValueError),
                        'authoriz|consum|duplicate|stale'):
                    getattr(restarted, getter_name)(full)

    def test_same_formal_wrapper_claims_before_concurrent_audit(self):
        full = [0, 1, 2, 3]

        class CountingLoader:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(((torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),))

            def __len__(self):
                return 1

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.authorize_formal_access(
                split='test', phase='final_test_post_install',
                event_idx=1, task_id=1,
                timeline_step='event_1_CIL', classes=full,
            )
            underlying = CountingLoader()
            with mock.patch.object(
                    dataset, '_loader', return_value=underlying):
                loader = dataset.get_test_loader(full)

            first_append = threading.Event()
            two_appends = threading.Event()
            release = threading.Event()
            calls_lock = threading.Lock()
            append_calls = 0
            original_append = dataset._append_formal_access_record

            def delayed_append(record):
                nonlocal append_calls
                with calls_lock:
                    append_calls += 1
                    if append_calls == 2:
                        two_appends.set()
                first_append.set()
                if not release.wait(timeout=5):
                    raise RuntimeError('timed out waiting for concurrent attempt')
                return original_append(record)

            outcomes = []

            def consume():
                try:
                    list(loader)
                    outcomes.append('success')
                except Exception as error:
                    outcomes.append(type(error).__name__)

            with mock.patch.object(
                    dataset, '_append_formal_access_record',
                    side_effect=delayed_append):
                first = threading.Thread(target=consume)
                second = threading.Thread(target=consume)
                first.start()
                self.assertTrue(first_append.wait(timeout=5))
                second.start()
                raced_into_audit = two_appends.wait(timeout=1)
                release.set()
                first.join(timeout=10)
                second.join(timeout=10)

            self.assertFalse(raced_into_audit)
            self.assertEqual(append_calls, 1)
            self.assertEqual(sorted(outcomes), ['RuntimeError', 'success'])
            self.assertEqual(underlying.iterations, 1)
            records = (
                Path(tmp) / 'data_flow_audit.jsonl'
            ).read_text().splitlines()
            self.assertEqual(len(records), 1)

    def test_formal_loader_rejects_tampered_durable_class_schema(self):
        full = [0, 1, 2, 3]
        access = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        mutations = {
            'missing': lambda record: record['access'].pop('classes'),
            'extra': lambda record: record['access'].__setitem__(
                'classes', [0, 1, 2, 3, 4]
            ),
            'order': lambda record: record['access'].__setitem__(
                'classes', [3, 2, 1, 0]
            ),
            'type': lambda record: record['access'].__setitem__(
                'classes', [0, 1, 2, '3']
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                dataset = self._formal_dataset(tmp)
                dataset.authorize_formal_access(**access)
                marker = Path(tmp) / 'formal_access' / 'FORMAL_ACCESS_PENDING.json'
                value = json.loads(marker.read_text())
                mutate(value)
                marker.chmod(0o644)
                marker.write_text(json.dumps(value, sort_keys=True))
                marker.chmod(0o444)
                with self.assertRaisesRegex(ValueError, 'durable|classes|schema'):
                    dataset.get_test_loader(full)

    def test_global_pending_survives_restart_and_only_exact_loader_consumes(self):
        full = [0, 1, 2, 3]
        validation = {
            'split': 'validation',
            'phase': 'final_validation_pre_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        test = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = self._formal_dataset(tmp)
            first.args.head_consolidation_enabled = 1
            first.authorize_formal_access(**validation)

            restarted = self._formal_dataset(tmp)
            restarted.args.head_consolidation_enabled = 1
            restarted.validation_indices = {0, 1, 2, 3}
            with self.assertRaisesRegex(RuntimeError, 'unconsumed|pending'):
                restarted.authorize_formal_access(**test)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed|pending'):
                restarted.authorize_formal_access(**validation)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed|pending'):
                restarted.get_train_loader([0, 1])
            self.assertEqual(len(restarted.get_train_loader([], shuffle=False)), 0)
            with self.assertRaisesRegex(ValueError, 'split|pending'):
                restarted.get_test_loader(full)

            list(restarted.get_validation_loader(full))
            restarted.authorize_formal_access(**test)
            list(restarted.get_test_loader(full))
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(
                [record['split'] for record in records],
                ['validation', 'test'],
            )

    def test_durable_history_enforces_phase_order_and_external_direct_test(self):
        full = [0, 1, 2, 3]
        validation = {
            'split': 'validation',
            'phase': 'final_validation_pre_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        test = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            internal = self._formal_dataset(tmp)
            internal.args.head_consolidation_enabled = 1
            with self.assertRaisesRegex(ValueError, 'validation|phase|order'):
                internal.authorize_formal_access(**test)

        with tempfile.TemporaryDirectory() as tmp:
            external = self._formal_dataset(tmp)
            external.authorize_formal_access(**test)
            list(external.get_test_loader(full))
            with self.assertRaisesRegex(ValueError, 'validation|phase|order'):
                external.authorize_formal_access(**validation)
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual([record['split'] for record in records], ['test'])

    def test_persisted_nonfinal_formal_access_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {
                'event': 'first_iteration',
                'loader_key': "('test', (0, 1, 2, 3))",
                'split': 'test', 'phase': 'final_test_post_install',
                'event_idx': 0, 'task_id': 0,
                'timeline_step': 'event_0_CIL',
                'classes': [0, 1, 2, 3],
            }
            (Path(tmp) / 'data_flow_audit.jsonl').write_text(
                json.dumps(record, sort_keys=True) + '\n'
            )
            dataset = self._formal_dataset(tmp)
            with self.assertRaisesRegex(ValueError, 'final|boundary|schema'):
                dataset.authorize_formal_access(
                    split='test', phase='final_test_post_install',
                    event_idx=1, task_id=1, timeline_step='event_1_CIL',
                    classes=[0, 1, 2, 3],
                )

    def test_rehydrated_pending_capability_revalidates_current_phase_protocol(self):
        full = [0, 1, 2, 3]
        test = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            external = self._formal_dataset(tmp)
            external.authorize_formal_access(**test)

            internal = self._formal_dataset(tmp)
            internal.args.head_consolidation_enabled = 1
            internal.validation_indices = {0, 1, 2, 3}
            with self.assertRaisesRegex(ValueError, 'validation|phase|order'):
                internal.get_test_loader(full)
            with self.assertRaisesRegex(ValueError, 'validation|phase|order'):
                internal.get_train_loader([0, 1])

        with tempfile.TemporaryDirectory() as tmp:
            external = self._formal_dataset(tmp)
            external.authorize_formal_access(**test)
            restarted = self._formal_dataset(tmp)
            list(restarted.get_test_loader(full))
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual([record['split'] for record in records], ['test'])

    def test_concurrent_restarted_consumers_append_exactly_one_access_row(self):
        full = [0, 1, 2, 3]
        test = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            owner = self._formal_dataset(tmp)
            owner.authorize_formal_access(**test)
            barrier = threading.Barrier(2)
            original = VFLDataset._claim_formal_pending
            outcomes = []

            def synchronized_claim(dataset, pending):
                barrier.wait(timeout=5)
                return original(dataset, pending)

            def consume():
                dataset = self._formal_dataset(tmp)
                try:
                    list(dataset.get_test_loader(full))
                    outcomes.append('success')
                except Exception as error:
                    outcomes.append(type(error).__name__)

            with mock.patch.object(
                    VFLDataset, '_claim_formal_pending', synchronized_claim):
                threads = [threading.Thread(target=consume) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
                self.assertTrue(all(not thread.is_alive() for thread in threads))

            self.assertEqual(outcomes.count('success'), 1)
            self.assertEqual(len(outcomes), 2)
            self.assertIn(
                next(outcome for outcome in outcomes if outcome != 'success'),
                ('RuntimeError', 'ValueError'),
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['split'], 'test')

    def test_consumption_crash_boundaries_fail_closed_or_recover_completed(self):
        full = [0, 1, 2, 3]
        test = {
            'split': 'test', 'phase': 'final_test_post_install',
            'event_idx': 1, 'task_id': 1,
            'timeline_step': 'event_1_CIL', 'classes': full,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.authorize_formal_access(**test)
            dataset.get_test_loader(full)
            consuming = Path(tmp) / 'formal_access' / 'FORMAL_ACCESS_CONSUMING.json'
            self.assertTrue(consuming.is_file())
            restarted = self._formal_dataset(tmp)
            with self.assertRaisesRegex(RuntimeError, 'consum|incomplete'):
                restarted.get_test_loader(full)
            self.assertFalse((Path(tmp) / 'data_flow_audit.jsonl').exists())

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._formal_dataset(tmp)
            dataset.authorize_formal_access(**test)
            loader = dataset.get_test_loader(full)
            import adaptive_consolidation_audit as audit_module
            original_write = audit_module.atomic_write_new_json

            def fail_completion(path, payload):
                if str(path).endswith('.consumed.json'):
                    raise RuntimeError('simulated completion crash')
                return original_write(path, payload)

            with mock.patch.object(
                    audit_module, 'atomic_write_new_json', side_effect=fail_completion):
                with self.assertRaisesRegex(RuntimeError, 'completion crash'):
                    list(loader)
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            restarted = self._formal_dataset(tmp)
            with self.assertRaisesRegex(RuntimeError, 'consum|incomplete'):
                restarted.get_test_loader(full)
            self.assertEqual(
                len((Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()),
                1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            validation = {
                'split': 'validation',
                'phase': 'final_validation_pre_install',
                'event_idx': 1, 'task_id': 1,
                'timeline_step': 'event_1_CIL', 'classes': full,
            }
            dataset = self._formal_dataset(tmp)
            dataset.args.head_consolidation_enabled = 1
            dataset.authorize_formal_access(**validation)
            loader = dataset.get_validation_loader(full)
            with mock.patch.object(
                    dataset, '_remove_formal_marker',
                    side_effect=RuntimeError('simulated cleanup crash')):
                with self.assertRaisesRegex(RuntimeError, 'cleanup crash'):
                    list(loader)
            restarted = self._formal_dataset(tmp)
            restarted.args.head_consolidation_enabled = 1
            restarted.authorize_formal_access(**test)
            list(restarted.get_test_loader(full))
            records = [
                json.loads(line)
                for line in (Path(tmp) / 'data_flow_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(
                [record['split'] for record in records],
                ['validation', 'test'],
            )


if __name__ == '__main__':
    unittest.main()
