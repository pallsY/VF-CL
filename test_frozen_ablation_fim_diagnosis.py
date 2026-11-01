import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

import frozen_ablation_gpu_admission as admission
try:
    import frozen_ablation_fim_diagnosis as diagnosis
except ImportError:
    diagnosis = None


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(diagnosis, 'numerical diagnosis wrapper is missing')

    def row(self, value=1.0):
        return dict(task=0, name='party0/weight', kind='mean', shape=[], dtype='torch.float32',
                    values=[value], selected=True, accumulated=True)

    def test_required_startup_environment_rejects_each_missing_or_wrong_value(self):
        required = dict(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', PYTHONHASHSEED='42')
        with patch.dict(os.environ, required, clear=True):
            diagnosis.require_environment()
        for name in required:
            for value in (None, 'wrong'):
                changed = dict(required)
                if value is None:
                    changed.pop(name)
                else:
                    changed[name] = value
                with self.subTest(name=name, value=value), patch.dict(
                        os.environ, changed, clear=True), self.assertRaisesRegex(ValueError, name):
                    diagnosis.require_environment()

    def test_missing_startup_environment_blocks_capture_and_replay_before_gpu_gate(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                admission, 'ALLOWED_ROOTS', (Path(temporary),)), patch.dict(os.environ, {}, clear=True), \
                patch.object(admission, 'wait_gate') as gate, patch.object(admission, 'run') as training, \
                patch.object(diagnosis, 'load_seal', return_value={}):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                diagnosis.capture(root / 'code', root / 'capture-output', root / 'old')
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                diagnosis.replay(root / 'capture', root / 'reference', root / 'old', root / 'replay-output')
            gate.assert_not_called()
            training.assert_not_called()

    @unittest.skipIf(os.name == 'nt', 'shell launcher is verified on Linux')
    def test_shell_launcher_sets_environment_before_cpu_only_check(self):
        launcher = Path(__file__).with_name('run_frozen_fim_diagnosis.sh')
        result = subprocess.run(['bash', str(launcher), sys.executable, 'check-environment'],
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG='wrong', OMP_NUM_THREADS='wrong',
                     MKL_NUM_THREADS='wrong', PYTHONHASHSEED='0'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_tolerance_cannot_hide_changed_selection_or_accumulated_mask(self):
        original = self.row()
        close = self.row(1.0 + 1e-7)
        self.assertEqual(diagnosis.compare_rows([close], [original])['status'],
                         'diagnosis_supported')
        for field in ('selected', 'accumulated'):
            changed = dict(close, **{field: False})
            result = diagnosis.compare_rows([changed], [original])
            self.assertEqual(result['status'], 'diagnosis_rejected')
            self.assertEqual(result['rows'][0]['outside_tolerance'], 0)

    def test_nonfinite_missing_shape_dtype_and_outside_tolerance_reject(self):
        original = self.row()
        for change in ({'values': [float('nan')]}, {'values': [float('inf')]},
                       {'shape': [1]}, {'dtype': 'torch.float64'}, {'values': []},
                       {'name': 'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnosis.compare_rows([dict(original, **change)], [original])
        with self.assertRaises(ValueError):
            diagnosis.compare_rows([], [original])
        self.assertEqual(diagnosis.compare_rows([self.row(1.01)], [original])['status'],
                         'diagnosis_rejected')
        for field in ('selected', 'accumulated'):
            missing = dict(original)
            missing.pop(field)
            with self.assertRaises(ValueError):
                diagnosis.compare_rows([missing], [missing])
            with self.assertRaises(ValueError):
                diagnosis.compare_rows([dict(original, **{field: 1})], [original])

    def test_capture_observes_actual_results_without_replacing_math(self):
        payload = dict(tensors=[], quantiles=[])
        observed = diagnosis.capture_observer(payload)
        def tracked_tensor_mean(self):
            fim_capture = dict(task=0, names=['party0/weight'])
            index = 0
            result = self.mean()
            observed(self, torch, np)
            return result
        raw = torch.arange(32, dtype=torch.float32).reshape(4, 8)[::2, ::2]
        mean = tracked_tensor_mean(raw)
        def quantile(values):
            args, kwargs, task = (.75,), {}, {'id': 0}
            result = torch.quantile(values, *args)
            observed(values, torch, np)
            return result
        threshold = quantile(torch.tensor([mean.item(), 8.0]))
        self.assertTrue(torch.equal(payload['tensors'][0]['tensor'], raw))
        self.assertEqual(payload['tensors'][0]['stride'], list(raw.stride()))
        self.assertTrue(torch.equal(payload['tensors'][0]['native_mean'], mean))
        self.assertTrue(torch.equal(payload['quantiles'][0]['threshold'], threshold))
        self.assertEqual(observed(raw, torch, np), diagnosis.HASH(raw, torch, np))

    def test_driver_version_must_reproduce_the_original_host_driver(self):
        diagnosis.require_driver({'driver_version': '535.230.02'},
                                 {'launch_recheck': {'driver_version': '535.230.02'}})
        for driver in ('580.178.04', None):
            with self.assertRaisesRegex(ValueError, 'driver'):
                diagnosis.require_driver({'driver_version': driver},
                    {'launch_recheck': {'driver_version': '535.230.02'}})

    def test_dirty_checkout_rejects_before_any_gpu_use(self):
        with patch('subprocess.check_output', return_value=' M frozen_ablation_manifest.py\n'), \
                patch.object(admission, 'query_gpu') as gpu:
            with self.assertRaisesRegex(ValueError, 'clean'):
                diagnosis.require_clean_checkout()
            gpu.assert_not_called()
        with patch('subprocess.check_output', return_value=''):
            diagnosis.require_clean_checkout()

    def test_replay_checks_capture_seal_before_any_gpu_or_dataset_use(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                admission, 'ALLOWED_ROOTS', (Path(temporary),)), patch.object(
                admission, 'query_gpu') as gpu, patch.object(diagnosis, 'load_seal',
                    side_effect=ValueError('seal differs')):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'seal'):
                diagnosis.replay(root / 'capture', root / 'ref', root / 'candidate', root / 'output')
            gpu.assert_not_called()
            report = json.loads((root / 'output/DIAGNOSIS_FAILURE.json').read_text())
            self.assertEqual(report['status'], 'diagnosis_rejected')

    def test_fresh_paths_and_seal_detect_changed_or_added_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                admission, 'ALLOWED_ROOTS', (Path(temporary),)):
            root = diagnosis.fresh_output(Path(temporary) / 'capture')
            with self.assertRaises(FileExistsError):
                diagnosis.fresh_output(root)
            admission.write(root / 'data.json', {'test_accessed': False})
            diagnosis.seal(root, {'test_accessed': False})
            diagnosis.load_seal(root)
            path = root / 'data.json'
            original = path.read_bytes()
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'seal'):
                diagnosis.load_seal(root)
            path.write_bytes(original)
            (root / 'extra').write_text('added')
            with self.assertRaisesRegex(ValueError, 'seal'):
                diagnosis.load_seal(root)

    def test_no_test_audit_protection_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                admission, 'ALLOWED_ROOTS', (Path(temporary),)):
            root = Path(temporary)
            (root / 'run/formal_access').mkdir(parents=True)
            (root / 'run/data_flow_audit.jsonl').write_text('{"split":"test"}\n')
            with self.assertRaisesRegex(ValueError, 'no-test'):
                diagnosis.check_no_test(root)

    def test_original_provenance_requires_all_non_score_evidence_exact(self):
        record = {field: [{'task': 0, 'value': 'same'}]
                  for field in admission.NUMERIC_FIELDS}
        record.update(orchestration_commit=diagnosis.HISTORICAL_PRODUCER,
                      source_config_sha256='a', runtime={'host': 'a', 'torch': 'x'})
        other = copy.deepcopy(record)
        other['runtime']['host'] = 'b'
        other['fim_scores'][0]['value'] = 'known original difference'
        diagnosis.compare_historical(record, other)
        for field in ('states', 'fim_tensor_values', 'gradient_steps',
                      'sample_orders', 'losses', 'applied_freeze',
                      'source_config_sha256', 'orchestration_commit', 'runtime'):
            altered = copy.deepcopy(other)
            altered[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnosis.compare_historical(record, altered)

    def fixture(self):
        payload = dict(tensors=[], quantiles=[])
        record = dict(fim_tensor_values=[], fim_scores=[], states=[])
        masks = [{} for _ in range(4)]
        for task in (0, 1):
            tensors = []
            for party in range(4):
                values = []
                for index, name in enumerate(('weight', 'bias')):
                    tensor = torch.tensor([[1., 3.], [5., 7.]]) * (index + 1 + task)
                    mean = tensor.mean()
                    values.append(mean.item())
                    full_name = f'party{party}/{name}'
                    payload['tensors'].append(dict(task=task, name=full_name, tensor=tensor,
                        native_mean=mean, shape=list(tensor.shape), stride=list(tensor.stride()),
                        dtype=str(tensor.dtype), device='cuda:0'))
                    tensors.append(dict(name=full_name, shape=list(tensor.shape),
                        dtype=str(tensor.dtype), sha256=diagnosis.HASH(tensor, torch, np)))
                vector = torch.tensor(values)
                threshold = torch.quantile(vector, .75)
                payload['quantiles'].append(dict(task=task, q=.75, values=vector,
                                                 threshold=threshold))
                record['fim_scores'].append(dict(task=task,
                    values_sha256=diagnosis.HASH(vector, torch, np),
                    threshold_sha256=diagnosis.HASH(threshold, torch, np)))
                for name, mean in zip(('weight', 'bias'), values):
                    masks[party][name] = masks[party].get(name, False) or mean >= threshold.item()
            record['fim_tensor_values'].append(dict(task=task, tensors=tensors))
            record['states'].append(dict(task=task, fim_masks_sha256=diagnosis.HASH(masks, torch, np)))
        return payload, record, dict(num_parties=4, fim_freeze_frac=.25)

    def test_payload_binds_raw_inputs_native_means_eight_scores_and_accumulated_masks(self):
        payload, record, config = self.fixture()
        rows = diagnosis.validate_payload(payload, record, config)
        self.assertEqual(len(rows), 24)
        for change in ('tensor', 'stride', 'mean', 'vector', 'threshold', 'q', 'mask', 'missing'):
            altered, old = copy.deepcopy(payload), copy.deepcopy(record)
            if change == 'tensor':
                altered['tensors'][0]['tensor'][0, 0] += 1
            elif change == 'stride':
                altered['tensors'][0]['stride'] = [1, 2]
            elif change == 'mean':
                altered['tensors'][0]['native_mean'] += 1
            elif change in ('vector', 'threshold'):
                key = 'values' if change == 'vector' else 'threshold'
                altered['quantiles'][0][key] += 1
            elif change == 'q':
                altered['quantiles'][0]['q'] = .5
            elif change == 'mask':
                old['states'][1]['fim_masks_sha256'] = '0' * 64
            else:
                altered['quantiles'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnosis.validate_payload(altered, old, config)

    def test_reference_reconstruction_uses_raw_layout_and_binds_historical_score_hashes(self):
        payload, record, config = self.fixture()
        rows, raw = diagnosis.reconstruct_reference(payload, record, config, 'cpu')
        self.assertEqual(len(raw['tensors']), 16)
        self.assertEqual(diagnosis.compare_rows(rows, diagnosis.validate_payload(payload, record, config))
                         ['status'], 'diagnosis_supported')
        changed = copy.deepcopy(record)
        changed['fim_scores'][0]['values_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            diagnosis.reconstruct_reference(payload, changed, config, 'cpu')


if __name__ == '__main__':
    unittest.main()
