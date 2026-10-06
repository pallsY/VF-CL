import copy
import unittest
import tempfile
import json
import io
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import torch

from party_drift_telemetry import summarize_party_drift
from cl_methods.proto_evolve import ProtoEvolveCL
from config import get_config


class PartyDriftTelemetryTests(unittest.TestCase):
    def test_tracks_party_rotation_on_same_training_replay_without_mutation(self):
        old = [torch.nn.Linear(2, 2, bias=False).eval() for _ in range(2)]
        new = [copy.deepcopy(model).eval() for model in old]
        with torch.no_grad():
            for model in old + new:
                model.weight.copy_(torch.eye(2))
            new[1].weight.copy_(torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        replay = {
            0: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            1: torch.tensor([[1.0, 1.0], [-1.0, 1.0]]),
        }
        weights = {0: [0.7, 0.3], 1: [0.2, 0.8]}
        before = [copy.deepcopy(model.state_dict()) for model in old + new]

        result = summarize_party_drift(old, new, replay, weights,
                                       lambda batch: [batch, batch])

        self.assertEqual(result['num_parties'], 2)
        self.assertEqual(result['total_examples'], 4)
        self.assertEqual(set(result['classes']), {'0', '1'})
        for class_id in ('0', '1'):
            row = result['classes'][class_id]
            self.assertEqual(row['sample_count'], 2)
            self.assertAlmostEqual(row['party_cosine_drift'][0], 0.0, places=6)
            self.assertAlmostEqual(row['party_cosine_drift'][1], 1.0, places=6)
            self.assertEqual(row['party_weights'], weights[int(class_id)])
        for model, state in zip(old + new, before):
            for name, value in state.items():
                self.assertTrue(torch.equal(model.state_dict()[name], value))

    def test_method_record_uses_only_old_class_replay(self):
        old = [torch.nn.Linear(2, 2, bias=False).eval() for _ in range(2)]
        current = [copy.deepcopy(model).eval() for model in old]
        with torch.no_grad():
            for model in old + current:
                model.weight.copy_(torch.eye(2))
            current[1].weight.copy_(torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        method.args = SimpleNamespace(num_parties=2, data='tabvfl',
                                      party_col_ranges=[(0, 2), (2, 4)],
                                      device='cpu', seed=42)
        method.trainer = SimpleNamespace(bottoms=current)
        method._old_bottoms = old
        method.current_task_classes = [1]
        method.head_raw_replay = {
            0: torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
            1: torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        }
        method.class_party_weights = {0: [0.5, 0.5]}
        method.global_protos = {0: {}, 1: {}}

        record = method._build_party_drift_record(1)

        self.assertEqual(record['task_id'], 1)
        self.assertEqual(record['old_class_ids'], [0])
        self.assertEqual(set(record['measurement']['classes']), {'0'})
        self.assertFalse(record['validation_used'])
        self.assertFalse(record['test_used'])

    def test_writer_is_idempotent_and_fails_on_changed_resume_record(self):
        with tempfile.TemporaryDirectory() as directory:
            method = ProtoEvolveCL.__new__(ProtoEvolveCL)
            method.args = SimpleNamespace(output_dir=directory)
            method.party_drift_telemetry_enabled = True
            method._build_party_drift_record = mock.Mock(return_value={
                'schema_version': 1, 'task_id': 1,
            })
            def child(root, name):
                path = Path(root) / name
                path.mkdir(exist_ok=True)
                return path
            with mock.patch('adaptive_consolidation_audit._ensure_child_dir',
                            side_effect=child), \
                 mock.patch('adaptive_consolidation_audit._safe_json',
                            side_effect=lambda path: json.loads(Path(path).read_text())), \
                 mock.patch('adaptive_consolidation_audit.atomic_write_new_json',
                            side_effect=lambda path, row: Path(path).write_text(
                                json.dumps(row))) as writer:
                first = method._write_party_drift_record(1)
                self.assertEqual(first, method._write_party_drift_record(1))
                self.assertEqual(writer.call_count, 1)
                method._build_party_drift_record.return_value = {
                    'schema_version': 1, 'task_id': 1, 'changed': True,
                }
                with self.assertRaisesRegex(ValueError, 'changed'):
                    method._write_party_drift_record(1)
                method.party_drift_telemetry_enabled = False
                self.assertIsNone(method._write_party_drift_record(1))
                self.assertEqual(writer.call_count, 1)

    def test_config_defaults_off_and_requires_existing_replay_and_contributions(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('sys.argv', [
                    'vfcl', '--results_dir', directory, '--exp_name', 'off',
                    '--cl_method', 'proto_evolve']):
                self.assertEqual(get_config().party_drift_telemetry, 0)
            with mock.patch('sys.argv', [
                    'vfcl', '--results_dir', directory, '--exp_name', 'on',
                    '--cl_method', 'proto_evolve', '--head_consolidation_enabled', '1',
                    '--dep_tracking_enabled', '1', '--party_drift_telemetry', '1',
                    '--unlearn_after_tasks', '99,99']):
                self.assertEqual(get_config().party_drift_telemetry, 1)
            with mock.patch('sys.argv', [
                    'vfcl', '--results_dir', directory, '--exp_name', 'ul_blocked',
                    '--cl_method', 'proto_evolve', '--head_consolidation_enabled', '1',
                    '--dep_tracking_enabled', '1', '--party_drift_telemetry', '1',
                    '--num_tasks', '2', '--unlearn_after_tasks', '1,99']):
                with mock.patch('sys.stderr', io.StringIO()):
                    with self.assertRaises(SystemExit):
                        get_config()
            with mock.patch('sys.argv', [
                    'vfcl', '--results_dir', directory, '--exp_name', 'invalid',
                    '--cl_method', 'proto_evolve', '--party_drift_telemetry', '1']):
                with mock.patch('sys.stderr', io.StringIO()):
                    with self.assertRaises(SystemExit):
                        get_config()

    def test_identical_zero_features_have_zero_drift(self):
        old = torch.nn.Linear(2, 2, bias=False).eval()
        current = copy.deepcopy(old).eval()
        with torch.no_grad():
            old.weight.zero_()
            current.weight.zero_()
        replay = {0: torch.tensor([[1.0, 0.0]])}
        result = summarize_party_drift([old], [current], replay,
                                       {0: [1.0]}, lambda batch: [batch])
        self.assertEqual(result['classes']['0']['party_cosine_drift'], [0.0])
        with torch.no_grad():
            current.weight.copy_(torch.eye(2))
        changed = summarize_party_drift([old], [current], replay,
                                        {0: [1.0]}, lambda batch: [batch])
        self.assertEqual(changed['classes']['0']['party_cosine_drift'], [1.0])

    def test_tiny_nonzero_vectors_are_not_treated_as_zero(self):
        old = torch.nn.Linear(1, 1, bias=False).double().eval()
        current = copy.deepcopy(old).eval()
        replay = {0: torch.ones(1, 1, dtype=torch.float64)}
        with torch.no_grad():
            old.weight.zero_()
            current.weight.fill_(1e-9)
        one_sided = summarize_party_drift(
            [old], [current], replay, {0: [1.0]}, lambda batch: [batch],
        )
        self.assertAlmostEqual(one_sided['classes']['0']['party_cosine_drift'][0],
                               1.0, places=6)
        with torch.no_grad():
            old.weight.fill_(1e-9)
            current.weight.fill_(-1e-9)
        opposite = summarize_party_drift(
            [old], [current], replay, {0: [1.0]}, lambda batch: [batch],
        )
        self.assertAlmostEqual(opposite['classes']['0']['party_cosine_drift'][0],
                               2.0, places=6)

    def test_large_finite_double_features_remain_finite(self):
        old = torch.nn.Linear(1, 1, bias=False).double().eval()
        current = copy.deepcopy(old).eval()
        with torch.no_grad():
            old.weight.fill_(1e40)
            current.weight.fill_(1e40)
        result = summarize_party_drift(
            [old], [current], {0: torch.ones(2, 1, dtype=torch.float64)},
            {0: [1.0]}, lambda batch: [batch],
        )
        self.assertEqual(result['classes']['0']['party_cosine_drift'], [0.0])

    def test_rejects_forgotten_or_missing_old_replay_before_forward(self):
        model = torch.nn.Identity().eval()
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        method.args = SimpleNamespace(num_parties=1, data='tabvfl',
                                      party_col_ranges=[(0, 2)], device='cpu', seed=42)
        method.trainer = SimpleNamespace(bottoms=[model])
        method._old_bottoms = [model]
        method.current_task_classes = [1]
        method.global_protos = {0: {}}
        method.head_raw_replay = {0: torch.ones(1, 2),
                                  1: torch.ones(1, 2),
                                  2: torch.ones(1, 2)}
        method.class_party_weights = {0: [1.0], 2: [1.0]}
        with self.assertRaisesRegex(ValueError, 'retained old-class replay'):
            method._build_party_drift_record(1)
        method.global_protos = {0: {}, 2: {}}
        method.head_raw_replay.pop(2)
        with self.assertRaisesRegex(ValueError, 'retained old-class replay'):
            method._build_party_drift_record(1)

    def test_rejects_missing_or_nonfinite_replay(self):
        model = torch.nn.Linear(2, 2, bias=False).eval()
        replay = {0: torch.tensor([[float('nan'), 1.0]])}
        with self.assertRaises(ValueError):
            summarize_party_drift([model], [model], replay, {0: [1.0]},
                                  lambda batch: [batch])
        with self.assertRaises(ValueError):
            summarize_party_drift([model], [model], {0: torch.ones(1, 2)},
                                  {}, lambda batch: [batch])


if __name__ == '__main__':
    unittest.main()
