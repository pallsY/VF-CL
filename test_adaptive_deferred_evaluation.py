import copy
import hashlib
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import runner
import adaptive_consolidation_audit as consolidation_audit
from adaptive_consolidation_audit import (
    _snapshot_manifest,
    _snapshot_record,
    _validate_formal_bic_bundle,
    _strict_load_trainer_state,
    _task_readouts,
    atomic_write_new_json,
    evaluate_deferred_cil_trajectory,
    evaluate_formal_deferred_trajectory,
    prepare_formal_deferred_evaluation,
    save_deferred_cil_snapshot,
    save_final_adaptive_checkpoint,
    trainer_state_sha256,
)
from cl_methods import get_cl_method
from metrics import MetricsTracker, cache_formal_batches
from models import TopModel
from runner import (
    _atomic_torch_save,
    _evaluate_formal_from_single_access,
    _load_resume_checkpoint,
    _publish_formal_deferred_result,
    _prepare_adaptive_provenance,
    _recover_adaptive_cil_snapshot,
    _recover_atomic_checkpoint_temps as _runner_recover_checkpoint_temps,
    _save_cil_checkpoint,
    _validate_resume_candidate_loadability,
    _validate_rng_checkpoint_state,
    _valid_tracker_checkpoint_state,
    run_oracle,
)


class _StateTrainer:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state or {'score': 0})

    def get_state(self):
        return {'bottoms': [], 'top_model': copy.deepcopy(self.state)}

    def load_state(self, state):
        self.state = copy.deepcopy(state['top_model'])

    def evaluate(self, loader):
        labels = torch.cat([labels for _, labels in loader])
        predictions = torch.tensor([
            int(label) if self.state['correct'].get(int(label), False)
            else int(self.state['fallback'])
            for label in labels
        ])
        top = getattr(self, 'top_model', None)
        if top is not None and bool(getattr(top, '_adaptive_enabled', False)):
            classes = [int(value) for value in top._adaptive_class_order]
            columns = {class_id: column for column, class_id in enumerate(classes)}
            probs = torch.zeros(labels.numel(), len(classes))
            probs[torch.arange(labels.numel()), torch.tensor([
                columns[int(prediction)] for prediction in predictions
            ])] = 1.0
        else:
            width = int(max(labels.max().item(), predictions.max().item())) + 1
            probs = torch.zeros(labels.numel(), width)
            probs[torch.arange(labels.numel()), predictions] = 1.0
        return float((predictions == labels).float().mean()), probs, labels


class _FormalStateTrainer(_StateTrainer):
    def __init__(self, state=None):
        super().__init__(state)
        self.top_model = SimpleNamespace()

    def evaluate(self, loader):
        accuracy, probabilities, labels = super().evaluate(loader)
        if probabilities.size(1) < 4:
            probabilities = torch.nn.functional.pad(
                probabilities, (0, 4 - probabilities.size(1))
            )
        return accuracy, probabilities, labels


class _FormalBiCTrainer(_StateTrainer):
    def __init__(self, state=None):
        super().__init__(state or {
            'correct': {class_id: True for class_id in range(20)},
            'fallback': 0,
        })

    def evaluate(self, loader):
        accuracy, probabilities, labels = super().evaluate(loader)
        if probabilities.size(1) < 20:
            probabilities = torch.nn.functional.pad(
                probabilities, (0, 20 - probabilities.size(1))
            )
        return accuracy, probabilities, labels

    def collect_logits(self, loader):
        inputs = torch.cat([batch_x for batch_x, _batch_y in loader]).float()
        labels = torch.cat([batch_y for _batch_x, batch_y in loader]).long()
        logits = torch.full((labels.numel(), 20), -2.0, dtype=torch.float32)
        logits[torch.arange(labels.numel()), labels] = 3.0
        logits += inputs.reshape(inputs.size(0), -1)[:, :1] * 1e-4
        return logits, labels


class _StrictStateTrainer:
    def __init__(self):
        self.bottoms = []
        self.top_model = TopModel(2, 2, cosine=False)

    def get_state(self):
        return {
            'bottoms': [],
            'top_model': copy.deepcopy(self.top_model.state_dict()),
        }

    def load_state(self, state):
        raise AssertionError('formal reload bypassed strict trainer validation')


class _CLState:
    def __init__(self, value=0):
        self.value = value

    def get_state(self):
        return {'value': self.value}

    def load_state(self, state):
        if type(state) is not dict or set(state) != {'value'}:
            raise ValueError('method structure mismatch')
        self.value = state['value']


class _ResumeTasks:
    def get_timeline(self):
        return [
            {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
            {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
        ]

    def advance_task(self, _task_id):
        pass

    def apply_unlearn(self, _classes):
        pass


def _recover_atomic_checkpoint_temps(directory, args):
    return _runner_recover_checkpoint_temps(
        directory, args,
        _StateTrainer({'weight': torch.tensor([0.0])}),
        _CLState(0), _ResumeTasks(), MetricsTracker(), mock.Mock(),
    )


class _Dataset:
    def __init__(self, order):
        self.order = order
        self.test_accesses = []

    def get_test_loader(self, classes):
        self.order.append(('test', tuple(classes)))
        self.test_accesses.append(tuple(classes))
        labels = torch.tensor(list(classes) * 2)
        return [(torch.zeros(labels.numel(), 1), labels)]


class DeferredEvaluationTests(unittest.TestCase):
    def _resume_args(self, output_dir):
        return SimpleNamespace(
            output_dir=output_dir, resume_run_dir=output_dir,
            save_task_checkpoints=3, num_tasks=2, num_parties=0,
            seed=7, data='toy', cl_method='er',
            head_consolidation_enabled=0, head_consolidation_mode='none',
        )

    def test_trainer_state_hash_is_order_stable_and_byte_sensitive(self):
        first = {'bottoms': [{'weight': torch.tensor([1.0, 2.0])}], 'top': {}}
        reordered = OrderedDict([
            ('top', OrderedDict()),
            ('bottoms', [OrderedDict([
                ('weight', torch.tensor([1.0, 2.0]))
            ])]),
        ])
        changed = copy.deepcopy(first)
        changed['bottoms'][0]['weight'][0] += 1
        self.assertEqual(trainer_state_sha256(first), trainer_state_sha256(reordered))
        self.assertNotEqual(trainer_state_sha256(first), trainer_state_sha256(changed))

    def test_formal_source_provenance_rejects_full_matrix_baseline_mutations(self):
        source = Path(__file__).resolve().parent
        selectors = (
            'cl_methods/adagauss.py', 'cl_methods/afc.py',
            'cl_methods/der_pp.py', 'cl_methods/er_ace.py',
            'cl_methods/ewc.py', 'cl_methods/lwf_wa.py',
            'cl_methods/proto_fedspace.py', 'cl_methods/target.py',
        )
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / 'baseline-worktree'
            subprocess.run([
                'git', '-C', str(source), 'worktree', 'add', '--detach',
                str(worktree), 'HEAD',
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                provenance = consolidation_audit._formal_source_provenance
                genuine = provenance(worktree)
                for logical in selectors:
                    with self.subTest(logical=logical):
                        path = worktree / logical
                        original = path.read_bytes()
                        try:
                            path.write_bytes(original + b'\n# baseline source mutation\n')
                            with self.assertRaises(ValueError) as rejected:
                                provenance(worktree)
                            self.assertIn(logical, str(rejected.exception))
                        finally:
                            path.write_bytes(original)
                        self.assertEqual(genuine, provenance(worktree))
            finally:
                subprocess.run([
                    'git', '-C', str(source), 'worktree', 'remove', '--force',
                    str(worktree),
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_formal_source_provenance_rejects_dirty_temporary_git_worktree(self):
        provenance = getattr(
            consolidation_audit, '_formal_source_provenance', None
        )
        self.assertTrue(callable(provenance), 'missing formal source provenance')
        source = Path(__file__).resolve().parent
        genuine = provenance(source)
        self.assertEqual(
            set(genuine), {'schema_version', 'source_commit', 'source_sha256'}
        )
        selectors = (
            'three_dataset_formal_runtime.py',
            'adaptive_tinyimagenet_heldout.py',
            'adaptive_dual_branch_validation.py',
        )
        from three_dataset_formal_audit import _SOURCE_INVENTORY
        self.assertEqual(
            tuple(_SOURCE_INVENTORY),
            tuple(consolidation_audit.FORMAL_SOURCE_FILES),
        )

        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / 'dirty-worktree'
            subprocess.run([
                'git', '-C', str(source), 'worktree', 'add', '--detach',
                str(worktree), 'HEAD',
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for logical in selectors:
                    with self.subTest(logical=logical):
                        path = worktree / logical
                        original = path.read_bytes()
                        try:
                            path.write_bytes(
                                original + b'\nDIRTY_RUNTIME_SELECTOR = True\n'
                            )
                            with self.assertRaisesRegex(
                                    ValueError, 'source|Git|commit'):
                                provenance(worktree)
                        finally:
                            path.write_bytes(original)
                current = provenance(worktree)
                self.assertTrue(
                    set(selectors).issubset(current['source_sha256'])
                )
            finally:
                subprocess.run([
                    'git', '-C', str(source), 'worktree', 'remove', '--force',
                    str(worktree),
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if worktree.exists():
                    shutil.rmtree(worktree)

    def test_formal_bic_fit_corpus_distinguishes_internal_validation_and_external_calibration(self):
        build = getattr(
            consolidation_audit, '_formal_bic_fit_corpus', None
        )
        validate = getattr(
            consolidation_audit, '_validate_formal_bic_fit_corpus', None
        )
        self.assertTrue(callable(build), 'missing formal BiC fit-corpus builder')
        self.assertTrue(callable(validate), 'missing formal BiC fit-corpus validator')
        cached = cache_formal_batches(((
            torch.zeros(4, 1), torch.tensor([0, 0, 1, 1]),
        ),))
        selection = {
            'validation_manifest_sha256': 'a' * 64,
            'calibration_manifest_sha256': 'b' * 64,
        }
        validation_manifest = {'sha256': 'a' * 64}
        calibration_audit = {'manifest_sha256': 'b' * 64}
        external = {'protocol': {
            'cl_method': 'finetune', 'head_consolidation_enabled': 0,
            'head_consolidation_mode': 'full_classifier',
        }}
        internal = {'protocol': {
            'cl_method': 'proto_evolve', 'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
        }}
        external_corpus = build(
            external, cached, calibration_audit,
            validation_manifest, selection,
        )
        internal_corpus = build(
            internal, cached, calibration_audit,
            validation_manifest, selection,
        )
        self.assertEqual(external_corpus['split'], 'calibration')
        self.assertEqual(external_corpus['manifest_sha256'], 'b' * 64)
        self.assertEqual(internal_corpus['split'], 'validation')
        self.assertEqual(internal_corpus['manifest_sha256'], 'a' * 64)
        self.assertNotEqual(external_corpus, internal_corpus)
        validate(external_corpus, external)
        validate(internal_corpus, internal)
        forged = dict(internal_corpus, split='calibration')
        with self.assertRaisesRegex(ValueError, 'fit corpus'):
            validate(forged, internal)

    def test_trainer_state_hash_tags_container_boundaries(self):
        split_mapping = {'a': [], 'b': 1}
        nested_list = {'a': ['b', 1]}
        self.assertNotEqual(
            trainer_state_sha256(split_mapping),
            trainer_state_sha256(nested_list),
        )

    def test_readouts_map_adaptive_probability_columns_to_global_classes(self):
        class AdaptiveTrainer:
            top_model = SimpleNamespace(
                _adaptive_enabled=torch.tensor(True),
                _adaptive_class_order=torch.tensor([1, 3]),
            )

            def evaluate(self, loader):
                return (
                    0.0,
                    torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                    torch.tensor([1, 3]),
                )

        class_il, task_il = _task_readouts(
            AdaptiveTrainer(), [object()], [1, 3]
        )
        self.assertEqual(class_il, 1.0)
        self.assertEqual(task_il, 1.0)

    def test_stage_saver_is_deterministic_and_never_accesses_validation_or_test(self):
        trainer = _StateTrainer({'correct': {0: True}, 'fallback': 0})
        cl_method = _CLState(3)
        for output in ('a', 'b'):
            pass
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_args = SimpleNamespace(output_dir=first, seed=42, data='toy',
                                         cl_method='proto', num_parties=0)
            second_args = SimpleNamespace(output_dir=second, seed=42, data='toy',
                                          cl_method='proto', num_parties=0)
            first_path = save_deferred_cil_snapshot(
                trainer, cl_method, first_args, 0, 0, {0: [0, 1]}
            )
            second_path = save_deferred_cil_snapshot(
                trainer, cl_method, second_args, 0, 0, {0: [0, 1]},
                protocol_kind='adaptive',
            )
            self.assertEqual(Path(first_path).read_bytes(), Path(second_path).read_bytes())
            payload = torch.load(first_path, weights_only=False)
            self.assertEqual(payload['introduced_classes'], [0, 1])
            self.assertNotIn('validation', repr(payload).lower())
            self.assertNotIn('test', repr(payload).lower())
            before = Path(first_path).stat()
            self.assertEqual(first_path, save_deferred_cil_snapshot(
                trainer, cl_method, first_args, 0, 0, {0: [0, 1]}
            ))
            after = Path(first_path).stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns),
                             (after.st_ino, after.st_mtime_ns))

    def test_final_checkpoint_duplicate_uses_one_pinned_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _StateTrainer({'correct': {0: True}, 'fallback': 0})
            trainer.top_model = SimpleNamespace(
                classifier=SimpleNamespace(in_features=1, out_features=1),
                cosine=False,
            )
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            provenance = {
                'source_version': 1, 'source_commit': 'a' * 40,
                'planned_sha256': 'b' * 64, 'launched_sha256': 'c' * 64,
            }
            path = save_final_adaptive_checkpoint(
                trainer, _CLState(), args, provenance
            )
            before = Path(path).stat()
            with mock.patch(
                'adaptive_consolidation_audit._safe_stat',
                side_effect=AssertionError('separate stat is forbidden'),
            ):
                self.assertEqual(path, save_final_adaptive_checkpoint(
                    trainer, _CLState(), args, provenance
                ))
            after = Path(path).stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns),
                             (after.st_ino, after.st_mtime_ns))

    def test_snapshot_manifest_orders_multi_digit_events_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            save_deferred_cil_snapshot(
                _StateTrainer({'correct': {0: True}, 'fallback': 0}),
                _CLState(), args, 2, 0, {0: [0]},
            )
            save_deferred_cil_snapshot(
                _StateTrainer({'correct': {1: True}, 'fallback': 1}),
                _CLState(), args, 10, 1, {0: [0], 1: [1]},
            )
            self.assertEqual(
                [record['event_idx'] for record in _snapshot_manifest(Path(tmp))],
                [2, 10],
            )

    def test_resume_checkpoint_recovers_missing_nondeterministic_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=tmp, device='cpu', num_parties=0, deterministic=0,
                head_consolidation_enabled=1,
                head_consolidation_mode='adaptive_dual_branch',
            )
            checkpoint = {
                'step': 'event_2_CIL', 'event_idx': 2, 'task_id': 1,
                'new_classes': [2, 3],
                'seen_task_classes': {0: [0, 1], 1: [2, 3]},
            }
            path = _recover_adaptive_cil_snapshot(
                checkpoint,
                _StateTrainer({'correct': {2: True, 3: True}, 'fallback': 2}),
                _CLState(), args,
            )
            self.assertEqual(Path(path).name, 'event_2_CIL.pt')
            self.assertTrue(Path(path).is_file())

    def test_resume_discards_torn_regular_checkpoint_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            temporary = directory / 'resume_latest.pt.tmp'
            temporary.write_bytes(b'torn')
            args = SimpleNamespace(resume_run_dir=tmp, output_dir=tmp)
            self.assertEqual(
                _load_resume_checkpoint(
                    args, object(), object(), object(), object(), object()
                ),
                (0, {}, []),
            )
            self.assertFalse(temporary.exists())

    def test_recovery_promotes_valid_newer_checkpoint_temp_over_old_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1], 'seen_task_classes': {0: [0], 1: [1]},
            })
            with temporary.open('wb') as handle:
                torch.save(newer, handle)
                handle.flush()
                os.fsync(handle.fileno())
            _recover_atomic_checkpoint_temps(directory, args)
            recovered = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(recovered['event_idx'], 1)
            self.assertFalse(temporary.exists())

    def test_recovery_keeps_old_target_when_newer_temp_schema_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save({
                'schema_version': 4,
                'step': 'event_1_CIL', 'event_idx': 1,
            }, temporary)
            with self.assertRaisesRegex(ValueError, 'schema'):
                _recover_atomic_checkpoint_temps(directory, args)
            recovered = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(recovered['event_idx'], 0)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_newer_checkpoint_with_forged_state_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            forged = torch.load(target, map_location='cpu', weights_only=True)
            forged.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1], 'seen_task_classes': {0: [0], 1: [1]},
                'trainer_state': {'bottoms': [], 'top_model': {}},
                'rng_state': {
                    'python': (),
                    'numpy': {
                        'bit_generator': 'x', 'keys': torch.tensor([]),
                        'position': 0, 'has_gauss': 0,
                        'cached_gaussian': 0.0,
                    },
                    'torch': torch.tensor([]), 'cuda': [],
                },
            })
            torch.save(forged, temporary)
            with self.assertRaisesRegex(ValueError, 'schema/state'):
                _recover_atomic_checkpoint_temps(directory, args)
            recovered = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(recovered['event_idx'], 0)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_semantically_wrong_trainer_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            forged = torch.load(target, map_location='cpu', weights_only=True)
            forged.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1], 'seen_task_classes': {0: [0], 1: [1]},
                'trainer_state': {
                    'bottoms': [],
                    'top_model': {'bogus': torch.tensor([0.0])},
                },
            })
            torch.save(forged, temporary)
            with self.assertRaisesRegex(ValueError, 'trainer structure'):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(
                torch.load(target, map_location='cpu', weights_only=True)[
                    'event_idx'
                ],
                0,
            )
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_type_correct_but_unloadable_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            forged = torch.load(target, map_location='cpu', weights_only=True)
            forged.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1], 'seen_task_classes': {0: [0], 1: [1]},
            })
            forged['rng_state']['numpy']['keys'] = torch.ones(
                1, dtype=torch.int64
            )
            forged['rng_state']['numpy']['position'] = 0
            forged['rng_state']['torch'] = torch.ones(1, dtype=torch.uint8)
            torch.save(forged, temporary)
            with self.assertRaisesRegex(ValueError, 'RNG state'):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(
                torch.load(target, map_location='cpu', weights_only=True)[
                    'event_idx'
                ],
                0,
            )
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_empty_tracker_or_method_state(self):
        for field, expected in (
                ('tracker_state', 'schema/state'),
                ('cl_state', 'method structure')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                args = self._resume_args(tmp)
                _save_cil_checkpoint(
                    _StateTrainer({'weight': torch.tensor([0.0])}),
                    _CLState(0), args, 'event_0_CIL', 0, [0], {0: [0]},
                )
                directory = Path(tmp) / 'checkpoints'
                target = directory / 'resume_latest.pt'
                temporary = directory / 'resume_latest.pt.tmp'
                forged = torch.load(
                    target, map_location='cpu', weights_only=True
                )
                forged.update({
                    'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                    'new_classes': [1],
                    'seen_task_classes': {0: [0], 1: [1]},
                    field: {},
                })
                torch.save(forged, temporary)
                with self.assertRaisesRegex(ValueError, expected):
                    _recover_atomic_checkpoint_temps(directory, args)
                self.assertEqual(
                    torch.load(target, map_location='cpu', weights_only=True)[
                        'event_idx'
                    ],
                    0,
                )
                self.assertTrue(temporary.exists())

    def test_recovery_rejects_malformed_tracker_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            forged = torch.load(target, map_location='cpu', weights_only=True)
            forged.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1], 'seen_task_classes': {0: [0], 1: [1]},
            })
            forged['tracker_state']['task_acc_history'] = [0]
            torch.save(forged, temporary)
            with self.assertRaisesRegex(ValueError, 'schema/state'):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(
                torch.load(target, map_location='cpu', weights_only=True)[
                    'event_idx'
                ],
                0,
            )
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_nonmapping_tracker_records(self):
        for field in ('ul_metrics', 'comm_stats', 'timing', 'step_results'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                args = self._resume_args(tmp)
                _save_cil_checkpoint(
                    _StateTrainer({'weight': torch.tensor([0.0])}),
                    _CLState(0), args, 'event_0_CIL', 0, [0], {0: [0]},
                )
                directory = Path(tmp) / 'checkpoints'
                target = directory / 'resume_latest.pt'
                temporary = directory / 'resume_latest.pt.tmp'
                forged = torch.load(
                    target, map_location='cpu', weights_only=True
                )
                forged.update({
                    'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                    'new_classes': [1],
                    'seen_task_classes': {0: [0], 1: [1]},
                })
                forged['tracker_state'][field] = [0]
                torch.save(forged, temporary)
                with self.assertRaisesRegex(ValueError, 'schema/state'):
                    _recover_atomic_checkpoint_temps(directory, args)
                self.assertEqual(
                    torch.load(target, map_location='cpu', weights_only=True)[
                        'event_idx'
                    ],
                    0,
                )
                self.assertTrue(temporary.exists())

    def test_recovery_fails_closed_on_conflicting_same_event_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([1.0])}), _CLState(1), args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            temporary = directory / 'resume_latest.pt.tmp'
            conflicting = torch.load(
                target, map_location='cpu', weights_only=True
            )
            conflicting['trainer_state']['top_model']['weight'] = \
                torch.tensor([2.0])
            torch.save(conflicting, temporary)
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertTrue(target.exists())
            self.assertTrue(temporary.exists())

    def test_resume_recovers_checkpoint_temp_before_provenance_tree_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            temporary = directory / 'resume_latest.pt.tmp'
            temporary.write_bytes(b'torn')
            args = SimpleNamespace(resume_run_dir=tmp, output_dir=tmp)
            with mock.patch(
                'runner.prepare_adaptive_run_provenance', return_value={'ok': True}
            ) as prepare:
                self.assertEqual(_prepare_adaptive_provenance(args), {'ok': True})
            prepare.assert_called_once_with(tmp)
            self.assertFalse(temporary.exists())

    def test_checkpoint_atomic_write_refuses_preexisting_temp_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            outside = Path(tmp) / 'outside'
            outside.write_bytes(b'unchanged')
            target = directory / 'resume_latest.pt'
            os.symlink(outside, Path(str(target) + '.tmp'))
            with self.assertRaises(FileExistsError):
                _atomic_torch_save({'value': 1}, target)
            self.assertEqual(outside.read_bytes(), b'unchanged')
            self.assertTrue(Path(str(target) + '.tmp').is_symlink())

    def test_checkpoint_atomic_write_loser_never_deletes_winner_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            target = directory / 'resume_latest.pt'
            active_temp = Path(str(target) + '.tmp')
            active_temp.write_bytes(b'active-writer')
            with self.assertRaises(FileExistsError):
                _atomic_torch_save({'value': 1}, target)
            self.assertEqual(active_temp.read_bytes(), b'active-writer')

    def test_adaptive_resume_rejects_missing_party_state_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            protocol = {
                'seed': 1, 'data': 'toy', 'cl_method': 'proto_evolve',
                'num_tasks': 2, 'num_parties': 1,
                'head_consolidation_enabled': 1,
                'head_consolidation_mode': 'adaptive_dual_branch',
            }
            checkpoint = {
                'schema_version': 4, 'step': 'event_0_CIL', 'event_idx': 0,
                'task_id': 0, 'new_classes': [0],
                'seen_task_classes': {0: [0]}, 'forgotten_classes': [],
                'trainer_state': {'bottoms': [], 'top_model': {}},
                'cl_state': {}, 'bic_state': None, 'bic_history': [],
                'tracker_state': {}, 'rng_state': {}, 'protocol': protocol,
            }
            torch.save(checkpoint, directory / 'resume_latest.pt')
            args = SimpleNamespace(
                resume_run_dir=tmp, output_dir=tmp, **protocol
            )
            with self.assertRaisesRegex(RuntimeError, 'schema/state'):
                _load_resume_checkpoint(
                    args, object(), object(), object(), object(), object()
                )

    def test_resume_rejects_out_of_range_task_before_live_state_mutation(self):
        class TaskProbe:
            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
                ]

            def advance_task(self, task_id):
                if task_id >= 2:
                    raise ValueError('task out of range')

            def apply_unlearn(self, _classes):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            trainer = _StateTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args,
                'event_0_CIL', 0, [999], {0: [999]},
            )
            trainer.state = {'weight': torch.tensor([-1.0])}
            with self.assertRaisesRegex(RuntimeError, 'task/class timeline'):
                _load_resume_checkpoint(
                    args, trainer, method, TaskProbe(), MetricsTracker(),
                    mock.Mock(),
                )
            torch.testing.assert_close(
                trainer.state['weight'], torch.tensor([-1.0])
            )

    def test_resume_rejects_conflicting_permanent_same_event_candidates(self):
        class TaskProbe:
            def advance_task(self, _task_id):
                pass

            def apply_unlearn(self, _classes):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            trainer = _StateTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            rolling = directory / 'resume_latest.pt'
            conflicting = torch.load(
                rolling, map_location='cpu', weights_only=True
            )
            conflicting['trainer_state']['top_model']['weight'] = torch.tensor(
                [2.0]
            )
            torch.save(conflicting, directory / 'event_0_CIL.pt')
            trainer.state = {'weight': torch.tensor([-1.0])}
            with self.assertRaisesRegex(ValueError, 'conflicting resume'):
                _load_resume_checkpoint(
                    args, trainer, method, TaskProbe(), MetricsTracker(),
                    mock.Mock(),
                )
            torch.testing.assert_close(
                trainer.state['weight'], torch.tensor([-1.0])
            )

    def test_resume_preserves_old_target_for_timeline_invalid_newer_temp(self):
        class TaskProbe:
            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
                ]

            def advance_task(self, _task_id):
                pass

            def apply_unlearn(self, _classes):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            trainer = _StateTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [999],
                'seen_task_classes': {0: [0], 1: [999]},
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises((ValueError, RuntimeError)):
                _load_resume_checkpoint(
                    args, trainer, method, TaskProbe(), MetricsTracker(),
                    mock.Mock(),
                )
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_resume_preserves_old_target_for_load_invalid_newer_temp(self):
        class RejectingTrainer(_StateTrainer):
            def load_state(self, state):
                if float(state['top_model']['weight'].item()) == 999.0:
                    raise ValueError('trainer state is not loadable')
                super().load_state(state)

        class TaskProbe:
            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
                ]

            def advance_task(self, _task_id):
                pass

            def apply_unlearn(self, _classes):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            trainer = RejectingTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args,
                'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
            })
            newer['trainer_state']['top_model']['weight'] = torch.tensor(
                [999.0]
            )
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises((ValueError, RuntimeError)):
                _load_resume_checkpoint(
                    args, trainer, method, TaskProbe(), MetricsTracker(),
                    mock.Mock(),
                )
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_missing_and_extra_tracker_record_fields(self):
        for mutation in ('missing_comm_field', 'extra_task_field'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                args = self._resume_args(tmp)
                tracker = MetricsTracker()
                tracker.record_task_accuracies(
                    'event_0_CIL', {'task_0': 0.5}, 0.5
                )
                tracker.record_comm('event_0_CIL', {
                    'comm_rounds': 1, 'megabytes_transmitted': 2.0,
                })
                tracker.record_timing('event_0_CIL', 1.0)
                tracker.record_ul_result('event_0_CIL', {
                    'forget_acc': 0.0, 'retain_acc': 0.5,
                    'mia_score': 0.5,
                })
                tracker.record_step({
                    'event_idx': 0, 'type': 'CIL', 'task_id': 0,
                    'evaluation_deferred': True, 'train_time': 1.0,
                    'comm': {
                        'comm_rounds': 1, 'megabytes_transmitted': 2.0,
                    },
                })
                _save_cil_checkpoint(
                    _StateTrainer({'weight': torch.tensor([0.0])}),
                    _CLState(0), args, 'event_0_CIL', 0, [0], {0: [0]},
                    tracker_state=tracker.to_dict(),
                )
                directory = Path(tmp) / 'checkpoints'
                target = directory / 'resume_latest.pt'
                original = target.read_bytes()
                newer = torch.load(
                    target, map_location='cpu', weights_only=True
                )
                newer.update({
                    'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                    'new_classes': [1],
                    'seen_task_classes': {0: [0], 1: [1]},
                })
                if mutation == 'missing_comm_field':
                    newer['tracker_state']['comm_stats'][0].pop(
                        'megabytes_transmitted'
                    )
                else:
                    newer['tracker_state']['task_acc_history'][0][
                        'unexpected'
                    ] = True
                temporary = directory / 'resume_latest.pt.tmp'
                torch.save(newer, temporary)
                with self.assertRaises(ValueError):
                    _recover_atomic_checkpoint_temps(directory, args)
                self.assertEqual(target.read_bytes(), original)
                self.assertTrue(temporary.exists())

    def test_recovery_rejects_incomplete_deferred_final_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0),
                args, 'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            tracker = MetricsTracker()
            tracker.record_task_accuracies(
                'event_0_CIL', {'task_0': 0.5}, 0.5
            )
            tracker.record_task_accuracies(
                'event_1_CIL', {'task_0': 0.4, 'task_1': 0.5}, 0.45
            )
            tracker.task_acc_matrix[-1]['deferred_final'] = True
            forged_tracker = tracker.to_dict()
            self.assertEqual(forged_tracker['cl_metrics']['BWT'], 0.0)
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'tracker_state': forged_tracker,
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_oracle_records_under_er_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0),
                args, 'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            tracker = MetricsTracker()
            tracker.record_task_accuracies(
                'event_0_CIL', {'task_0': 0.5}, 0.5,
                per_task_debiased={'task_0': 0.5},
                per_task_taskil={'task_0': 0.5},
            )
            tracker.record_step({
                'event_idx': 0, 'type': 'CIL', 'overall_acc': 0.5,
                'per_task': {'task_0': 0.5},
                'per_task_debiased': {'task_0': 0.5},
                'per_task_taskil': {'task_0': 0.5},
            })
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'tracker_state': tracker.to_dict(),
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_tracker_schema_accepts_authentic_deferred_and_oracle_rows(self):
        deferred = MetricsTracker()
        deferred.record_task_accuracies(
            'event_0_CIL', {'task_0': 0.5}, 0.5
        )
        deferred.task_acc_matrix[-1]['deferred_diagonal'] = {'task_0': 0.5}
        deferred.record_task_accuracies(
            'event_2_CIL', {'task_0': 0.4, 'task_1': 0.6}, 0.5,
            per_task_taskil={'task_0': 0.7, 'task_1': 0.8},
        )
        deferred.task_acc_matrix[-1].update({
            'deferred_diagonal': {'task_0': 0.5, 'task_1': 0.6},
            'deferred_final': True,
            'deferred_final_task': 'task_1',
        })
        deferred_comm = {
            'comm_rounds': 1, 'megabytes_transmitted': 2.0,
        }
        for step in ('event_0_CIL', 'event_1_UL', 'event_2_CIL'):
            deferred.record_timing(step, 1.0)
            deferred.record_comm(step, deferred_comm)
        deferred.record_step({
            'event_idx': 0, 'type': 'CIL', 'task_id': 0,
            'evaluation_deferred': True, 'train_time': 1.0,
            'comm': deferred_comm,
        })
        deferred.record_step({
            'event_idx': 1, 'type': 'UL', 'forget_classes': [0],
            'evaluation_deferred': True, 'comm': deferred_comm,
        })
        deferred.record_step({
            'event_idx': 2, 'type': 'CIL', 'task_id': 1,
            'evaluation_deferred': True, 'train_time': 1.0,
            'comm': deferred_comm,
        })
        self.assertTrue(_valid_tracker_checkpoint_state(
            deferred.to_dict(), {
                'cl_method': 'proto_evolve',
                'num_tasks': 2,
                'head_consolidation_enabled': 1,
                'head_consolidation_mode': 'adaptive_dual_branch',
            },
        ))

        oracle = MetricsTracker()
        oracle.record_task_accuracies(
            'event_0_CIL', {'task_0': 0.5}, 0.5,
            per_task_debiased={'task_0': 0.5},
            per_task_taskil={'task_0': 0.5},
        )
        oracle.record_timing('event_0_CIL', 1.0)
        oracle.record_comm('event_0_CIL', {
            'comm_rounds': 1, 'megabytes_transmitted': 2.0,
        })
        oracle.record_step({
            'event_idx': 0, 'type': 'CIL', 'overall_acc': 0.5,
            'per_task': {'task_0': 0.5},
            'per_task_debiased': {'task_0': 0.5},
            'per_task_taskil': {'task_0': 0.5},
        })
        self.assertTrue(_valid_tracker_checkpoint_state(
            oracle.to_dict(), {
                'cl_method': 'oracle',
                'head_consolidation_enabled': 0,
                'head_consolidation_mode': 'full_classifier',
            },
        ))

    def test_tracker_new_classes_field_is_accepted_only_for_formal_deferred_cil(self):
        comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
        legacy = MetricsTracker()
        legacy.record_timing('event_0_CIL', 0.25)
        legacy.record_comm('event_0_CIL', comm)
        legacy.record_step({
            'event_idx': 0, 'type': 'CIL', 'task_id': 0,
            'evaluation_deferred': True, 'train_time': 0.25, 'comm': comm,
        })
        adaptive_protocol = {
            'cl_method': 'proto_evolve', 'num_tasks': 1,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
        }
        self.assertTrue(_valid_tracker_checkpoint_state(
            legacy.to_dict(), adaptive_protocol
        ))
        widened_legacy = legacy.to_dict()
        widened_legacy['step_results'][0]['new_classes'] = [0, 1]
        self.assertFalse(_valid_tracker_checkpoint_state(
            widened_legacy, adaptive_protocol
        ))

        formal_protocol = {
            'cl_method': 'finetune', 'num_tasks': 1,
            'head_consolidation_enabled': 0,
            'head_consolidation_mode': 'full_classifier',
            'formal_deferred_evaluation': True,
        }
        formal = copy.deepcopy(legacy.to_dict())
        formal['step_results'][0]['new_classes'] = [0, 1]
        self.assertTrue(_valid_tracker_checkpoint_state(
            formal, formal_protocol
        ))
        missing_formal_classes = copy.deepcopy(formal)
        del missing_formal_classes['step_results'][0]['new_classes']
        self.assertFalse(_valid_tracker_checkpoint_state(
            missing_formal_classes, formal_protocol
        ))

    def test_oracle_entry_establishes_its_actual_tracker_protocol(self):
        args = SimpleNamespace(
            cl_method='proto_evolve', ul_method='none',
            head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )
        with mock.patch(
            'runner.initialize_experiment_rng',
            side_effect=RuntimeError('stop before data access'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'stop before data access'):
                run_oracle(args)
        self.assertEqual(args.cl_method, 'oracle')
        self.assertEqual(args.ul_method, 'oracle')
        self.assertEqual(args.head_consolidation_enabled, 0)
        self.assertEqual(args.head_consolidation_mode, 'full_classifier')

    def test_tracker_schema_rejects_cross_protocol_and_inconsistent_deferred_rows(self):
        tracker = MetricsTracker()
        tracker.record_task_accuracies(
            'event_0_CIL', {'task_0': 0.5}, 0.5
        )
        tracker.task_acc_matrix[-1]['deferred_diagonal'] = {'task_0': 0.5}
        tracker.record_task_accuracies(
            'event_2_CIL', {'task_0': 0.4, 'task_1': 0.6}, 0.5,
            per_task_taskil={'task_0': 0.7, 'task_1': 0.8},
        )
        tracker.task_acc_matrix[-1].update({
            'deferred_diagonal': {'task_0': 0.5, 'task_1': 0.6},
            'deferred_final': True,
            'deferred_final_task': 'task_1',
        })
        state = tracker.to_dict()
        er_protocol = {
            'cl_method': 'er', 'num_tasks': 2,
            'head_consolidation_enabled': 0,
            'head_consolidation_mode': 'none',
        }
        adaptive_protocol = {
            'cl_method': 'proto_evolve', 'num_tasks': 2,
            'head_consolidation_enabled': 1,
            'head_consolidation_mode': 'adaptive_dual_branch',
        }
        self.assertFalse(_valid_tracker_checkpoint_state(state, er_protocol))
        wrong_maps = copy.deepcopy(state)
        wrong_maps['task_acc_history'][-1]['per_task_accs_taskil'] = {
            'task_9': 0.8,
        }
        self.assertFalse(_valid_tracker_checkpoint_state(
            wrong_maps, adaptive_protocol
        ))
        wrong_diagonal = copy.deepcopy(state)
        wrong_diagonal['task_acc_history'][0]['deferred_diagonal'] = {
            'task_0': 0.4,
        }
        self.assertFalse(_valid_tracker_checkpoint_state(
            wrong_diagonal, adaptive_protocol
        ))
        wrong_final_task = copy.deepcopy(state)
        wrong_final_task['task_acc_history'][-1]['deferred_final_task'] = \
            'task_0'
        self.assertFalse(_valid_tracker_checkpoint_state(
            wrong_final_task, adaptive_protocol
        ))

    def test_recovery_rejects_task_row_relabelled_away_from_its_cil_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0),
                args, 'event_0_CIL', 0, [0], {0: [0]},
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            tracker = MetricsTracker()
            tracker.record_task_accuracies(
                'event_0_CIL', {'task_0': 0.5}, 0.5,
                per_task_debiased={'task_0': 0.5},
                per_task_taskil={'task_0': 0.5},
            )
            comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
            tracker.record_timing('event_0_CIL', 1.0)
            tracker.record_comm('event_0_CIL', comm)
            tracker.record_step({
                'event_idx': 0, 'type': 'CIL', 'task_id': 0,
                'per_task_acc': {'task_0': 0.5},
                'per_task_acc_debiased': {'task_0': 0.5},
                'per_task_acc_taskil': {'task_0': 0.5},
                'overall_acc': 0.5, 'companion_readouts': {},
                'train_time': 1.0, 'comm': comm,
            })
            forged = tracker.to_dict()
            forged['task_acc_history'][0]['step'] = 'event_0_UL'
            round_trip = MetricsTracker()
            round_trip.load_dict(forged)
            forged = round_trip.to_dict()
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'tracker_state': forged,
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_self_consistent_rewritten_producer_timeline(self):
        def record_cil(tracker, event_idx, task_id, per_task):
            step = f'event_{event_idx}_CIL'
            comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
            tracker.record_task_accuracies(
                step, per_task, 0.5,
                per_task_debiased=per_task,
                per_task_taskil=per_task,
            )
            tracker.record_timing(step, 1.0)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': event_idx, 'type': 'CIL', 'task_id': task_id,
                'per_task_acc': per_task,
                'per_task_acc_debiased': per_task,
                'per_task_acc_taskil': per_task,
                'overall_acc': 0.5, 'companion_readouts': {},
                'train_time': 1.0, 'comm': comm,
            })

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            args.head_consolidation_mode = 'full_classifier'
            target_tracker = MetricsTracker()
            record_cil(target_tracker, 0, 0, {'task_0': 0.5})
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0),
                args, 'event_0_CIL', 0, [0], {0: [0]},
                tracker_state=target_tracker.to_dict(),
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            forged_tracker = MetricsTracker()
            record_cil(forged_tracker, 0, 0, {'task_0': 0.5})
            record_cil(
                forged_tracker, 42, 1, {'task_0': 0.4, 'task_1': 0.6}
            )
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'tracker_state': forged_tracker.to_dict(),
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_self_consistent_wrong_task_identity_maps(self):
        def record_cil(tracker, event_idx, task_id, per_task):
            step = f'event_{event_idx}_CIL'
            comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
            tracker.record_task_accuracies(
                step, per_task, 0.5,
                per_task_debiased=per_task, per_task_taskil=per_task,
            )
            tracker.record_timing(step, 1.0)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': event_idx, 'type': 'CIL', 'task_id': task_id,
                'per_task_acc': per_task,
                'per_task_acc_debiased': per_task,
                'per_task_acc_taskil': per_task,
                'overall_acc': 0.5, 'companion_readouts': {},
                'train_time': 1.0, 'comm': comm,
            })

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            args.head_consolidation_mode = 'full_classifier'
            target_tracker = MetricsTracker()
            record_cil(target_tracker, 0, 0, {'task_0': 0.5})
            _save_cil_checkpoint(
                _StateTrainer({'weight': torch.tensor([0.0])}), _CLState(0),
                args, 'event_0_CIL', 0, [0], {0: [0]},
                tracker_state=target_tracker.to_dict(),
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            forged_tracker = MetricsTracker()
            record_cil(forged_tracker, 0, 0, {'task_99': 0.5})
            record_cil(
                forged_tracker, 1, 1, {'task_0': 0.4, 'task_1': 0.6}
            )
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_1_CIL', 'event_idx': 1, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'tracker_state': forged_tracker.to_dict(),
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _recover_atomic_checkpoint_temps(directory, args)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_recovery_rejects_wrong_ul_attack_class_identity(self):
        class Timeline:
            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0]},
                    {'type': 'UL', 'after_task': 0, 'forget_classes': [0]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
                ]

            def advance_task(self, _task_id):
                pass

            def apply_unlearn(self, _classes):
                pass

        def record_cil(tracker, event_idx, task_id, per_task):
            step = f'event_{event_idx}_CIL'
            comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
            tracker.record_task_accuracies(
                step, per_task, 0.5,
                per_task_debiased=per_task, per_task_taskil=per_task,
            )
            tracker.record_timing(step, 1.0)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': event_idx, 'type': 'CIL', 'task_id': task_id,
                'per_task_acc': per_task,
                'per_task_acc_debiased': per_task,
                'per_task_acc_taskil': per_task,
                'overall_acc': 0.5, 'companion_readouts': {},
                'train_time': 1.0, 'comm': comm,
            })

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            args.head_consolidation_mode = 'full_classifier'
            target_tracker = MetricsTracker()
            record_cil(target_tracker, 0, 0, {'task_0': 0.5})
            trainer = _StateTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args, 'event_0_CIL', 0, [0], {0: [0]},
                tracker_state=target_tracker.to_dict(),
            )
            directory = Path(tmp) / 'checkpoints'
            target = directory / 'resume_latest.pt'
            original = target.read_bytes()
            forged_tracker = MetricsTracker()
            record_cil(forged_tracker, 0, 0, {'task_0': 0.5})
            ul_eval = {
                'forget_acc': 0.0, 'retain_acc': 0.0, 'mia_score': 0.5,
                'parties_touched': 0, 'parties_total': 0,
                'relearn_auc': {99: float('nan')},
                'reconnect_auc': {99: float('nan')},
            }
            ul_comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
            forged_tracker.record_task_accuracies(
                'event_1_UL', {}, 0.0,
                per_task_debiased={}, per_task_taskil={},
            )
            forged_tracker.record_ul_result('event_1_UL', ul_eval)
            forged_tracker.record_timing('event_1_UL', 1.0)
            forged_tracker.record_comm('event_1_UL', ul_comm)
            forged_tracker.record_step({
                'event_idx': 1, 'type': 'UL', 'forget_classes': [0],
                'ul_eval': ul_eval, 'per_task_acc': {},
                'per_task_acc_debiased': {}, 'per_task_acc_taskil': {},
                'overall_acc': 0.0,
            })
            record_cil(forged_tracker, 2, 1, {'task_1': 0.5})
            newer = torch.load(target, map_location='cpu', weights_only=True)
            newer.update({
                'step': 'event_2_CIL', 'event_idx': 2, 'task_id': 1,
                'new_classes': [1],
                'seen_task_classes': {0: [0], 1: [1]},
                'forgotten_classes': [0],
                'tracker_state': forged_tracker.to_dict(),
            })
            temporary = directory / 'resume_latest.pt.tmp'
            torch.save(newer, temporary)
            with self.assertRaises(ValueError):
                _runner_recover_checkpoint_temps(
                    directory, args, trainer, method, Timeline(),
                    MetricsTracker(), mock.Mock(),
                )
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(temporary.exists())

    def test_tracker_schema_rejects_divergent_producer_task_maps(self):
        tracker = MetricsTracker()
        tracker.record_task_accuracies(
            'event_0_CIL', {'task_0': 0.5}, 0.5,
            per_task_debiased={'task_99': 0.5},
            per_task_taskil={'task_99': 0.5},
        )
        comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
        tracker.record_timing('event_0_CIL', 1.0)
        tracker.record_comm('event_0_CIL', comm)
        tracker.record_step({
            'event_idx': 0, 'type': 'CIL', 'task_id': 0,
            'per_task_acc': {'task_0': 0.5},
            'per_task_acc_debiased': {'task_99': 0.5},
            'per_task_acc_taskil': {'task_99': 0.5},
            'overall_acc': 0.5, 'companion_readouts': {},
            'train_time': 1.0, 'comm': comm,
        })
        self.assertFalse(_valid_tracker_checkpoint_state(
            tracker.to_dict(), {
                'cl_method': 'er', 'num_tasks': 1,
                'head_consolidation_enabled': 0,
                'head_consolidation_mode': 'full_classifier',
            },
        ))

    def test_tracker_schema_rejects_evaluated_rows_in_adaptive_stream(self):
        tracker = MetricsTracker()
        per_task = {'task_0': 0.5}
        comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}
        tracker.record_task_accuracies(
            'event_0_CIL', per_task, 0.5,
            per_task_debiased=per_task, per_task_taskil=per_task,
        )
        tracker.record_timing('event_0_CIL', 1.0)
        tracker.record_comm('event_0_CIL', comm)
        tracker.record_step({
            'event_idx': 0, 'type': 'CIL', 'task_id': 0,
            'per_task_acc': per_task, 'per_task_acc_debiased': per_task,
            'per_task_acc_taskil': per_task, 'overall_acc': 0.5,
            'companion_readouts': {}, 'train_time': 1.0, 'comm': comm,
        })
        self.assertFalse(_valid_tracker_checkpoint_state(
            tracker.to_dict(), {
                'cl_method': 'proto_evolve', 'num_tasks': 1,
                'head_consolidation_enabled': 1,
                'head_consolidation_mode': 'adaptive_dual_branch',
            },
        ))

    def test_resume_accepts_authentic_post_ul_tracker_payload(self):
        class TaskProbe:
            def get_timeline(self):
                return [
                    {'type': 'CIL', 'task_id': 0, 'new_classes': [0, 2]},
                    {'type': 'UL', 'after_task': 0, 'forget_classes': [0]},
                    {'type': 'CIL', 'task_id': 1, 'new_classes': [1]},
                ]

            def advance_task(self, _task_id):
                pass

            def apply_unlearn(self, _classes):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            args = self._resume_args(tmp)
            args.head_consolidation_mode = 'full_classifier'
            tracker = MetricsTracker()
            comm = {'comm_rounds': 1, 'megabytes_transmitted': 2.0}

            def record_cil(event_idx, task_id, per_task):
                step = f'event_{event_idx}_CIL'
                tracker.record_task_accuracies(
                    step, per_task, 0.5,
                    per_task_debiased=per_task, per_task_taskil=per_task,
                )
                tracker.record_timing(step, 1.0)
                tracker.record_comm(step, comm)
                tracker.record_step({
                    'event_idx': event_idx, 'type': 'CIL', 'task_id': task_id,
                    'per_task_acc': per_task,
                    'per_task_acc_debiased': per_task,
                    'per_task_acc_taskil': per_task,
                    'overall_acc': 0.5, 'companion_readouts': {},
                    'train_time': 1.0, 'comm': comm,
                })

            record_cil(0, 0, {'task_0': 0.5})
            tracker.record_task_accuracies(
                'event_1_UL', {'task_0': 0.0}, 0.0,
                per_task_debiased={'task_0': 0.0},
                per_task_taskil={'task_0': 0.0},
            )
            ul_eval = {
                'forget_acc': 0.0, 'retain_acc': 0.5, 'mia_score': 0.5,
                'parties_touched': 1, 'parties_total': 2,
                'relearn_auc': {0: float('nan')},
                'reconnect_auc': {0: float('nan')},
            }
            tracker.record_ul_result('event_1_UL', ul_eval)
            tracker.record_timing('event_1_UL', 1.0)
            tracker.record_comm('event_1_UL', comm)
            tracker.record_step({
                'event_idx': 1, 'type': 'UL', 'forget_classes': [0],
                'ul_eval': ul_eval, 'per_task_acc': {'task_0': 0.0},
                'per_task_acc_debiased': {'task_0': 0.0},
                'per_task_acc_taskil': {'task_0': 0.0},
                'overall_acc': 0.0,
            })
            record_cil(2, 1, {'task_0': 0.5, 'task_1': 0.5})
            trainer = _StateTrainer({'weight': torch.tensor([0.0])})
            method = _CLState(0)
            _save_cil_checkpoint(
                trainer, method, args,
                'event_2_CIL', 1, [1], {0: [0, 2], 1: [1]},
                tracker_state=tracker.to_dict(), forgotten_classes=[0],
            )
            restored = MetricsTracker()
            start, seen, _ = _load_resume_checkpoint(
                args, trainer, method, TaskProbe(), restored, mock.Mock(),
            )
            self.assertEqual(start, 3)
            self.assertEqual(seen, {0: [0, 2], 1: [1]})
            restored_ul = restored.ul_metrics[0]
            self.assertEqual(
                set(restored_ul), {'step', *ul_eval}
            )
            self.assertTrue(math.isnan(restored_ul['relearn_auc'][0]))
            self.assertTrue(math.isnan(restored_ul['reconnect_auc'][0]))

    def test_evaluation_waits_for_audited_freeze_and_uses_isolated_sparse_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            stage_states = [
                {'correct': {0: True, 1: True}, 'fallback': 0},
                {'correct': {2: True, 3: True}, 'fallback': 0},
            ]
            paths = []
            for event_idx, state in enumerate(stage_states):
                trainer = _StateTrainer(state)
                paths.append(save_deferred_cil_snapshot(
                    trainer, _CLState(), args, event_idx, event_idx,
                    {0: [0, 1], 1: [2, 3]} if event_idx else {0: [0, 1]},
                ))
            final = run / 'adaptive_final.pt'
            torch.save({
                'schema_version': 1,
                'protocol': {'device': 'cpu', 'num_parties': 0},
                'trainer_state': {'bottoms': [], 'top_model': {
                    'correct': {0: False, 1: True, 2: True, 3: True},
                    'fallback': 1,
                }},
            }, final)
            freeze = {
                'status': 'ADAPTIVE_STATE_FROZEN',
                'audit_spec': {'checkpoint': final.name},
                'checkpoint': {
                    'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
                },
                'snapshots': [
                    _snapshot_record(path, run)[1] for path in paths
                ],
            }
            atomic_write_new_json(run / 'ADAPTIVE_STATE_FROZEN.json', freeze)
            order = []
            dataset = _Dataset(order)

            def audited(run_dir, spec):
                order.append(('audit', spec['checkpoint']))
                return copy.deepcopy(freeze)

            with mock.patch(
                'adaptive_consolidation_audit.audit_adaptive_checkpoint',
                side_effect=audited,
            ), mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                side_effect=lambda payload, _args: _StateTrainer(),
            ):
                result = evaluate_deferred_cil_trajectory(
                    paths, final, dataset, {0: [0, 1], 1: [2, 3]}, args
                )

            self.assertEqual(order[0], ('audit', final.name))
            self.assertEqual(dataset.test_accesses, [(0, 1), (2, 3), (0, 1), (2, 3)])
            self.assertEqual(set(result), {
                'cl_metrics', 'task_acc_history', 'ul_metrics',
                'comm_stats', 'timing', 'step_results',
            })
            history = result['task_acc_history']
            self.assertEqual(history[0]['per_task_accs'], {'task_0': 1.0})
            self.assertEqual(history[1]['per_task_accs'], {'task_0': 0.5, 'task_1': 1.0})
            self.assertEqual(history[1]['per_task_accs_taskil'], {'task_0': 0.5, 'task_1': 1.0})
            self.assertEqual(result['cl_metrics']['AA_final'], 0.75)
            self.assertEqual(result['cl_metrics']['BWT'], -0.5)

    def test_ul_filters_final_classes_and_preserves_cil_event_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            paths = [
                save_deferred_cil_snapshot(
                    _StateTrainer({'correct': {0: True, 1: True}, 'fallback': 0}),
                    _CLState(), args, 0, 0, {0: [0, 1]},
                ),
                save_deferred_cil_snapshot(
                    _StateTrainer({'correct': {2: True, 3: True}, 'fallback': 2}),
                    _CLState(), args, 2, 1, {0: [0, 1], 1: [2, 3]},
                ),
            ]
            final = run / 'adaptive_final.pt'
            torch.save({
                'schema_version': 1,
                'protocol': {'device': 'cpu', 'num_parties': 0},
                'trainer_state': {'bottoms': [], 'top_model': {
                    'correct': {1: True, 2: True, 3: True}, 'fallback': 1,
                }},
            }, final)
            freeze = {
                'status': 'ADAPTIVE_STATE_FROZEN',
                'audit_spec': {'checkpoint': final.name},
                'checkpoint': {
                    'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
                },
                'snapshots': [_snapshot_record(path, run)[1] for path in paths],
            }
            atomic_write_new_json(run / 'ADAPTIVE_STATE_FROZEN.json', freeze)
            dataset = _Dataset([])

            def fresh(payload, _args):
                trainer = _StateTrainer()
                if payload.get('kind') != 'adaptive_deferred_cil_snapshot':
                    trainer.top_model = SimpleNamespace(
                        _adaptive_enabled=torch.tensor(True),
                        _adaptive_class_order=torch.tensor([1, 2, 3]),
                    )
                return trainer

            with mock.patch(
                'adaptive_consolidation_audit.audit_adaptive_checkpoint',
                return_value=copy.deepcopy(freeze),
            ), mock.patch(
                'adaptive_consolidation_audit._fresh_trainer', side_effect=fresh,
            ):
                result = evaluate_deferred_cil_trajectory(
                    paths, final, dataset, {0: [0, 1], 1: [2, 3]}, args
                )

            self.assertEqual(
                dataset.test_accesses, [(0, 1), (2, 3), (1,), (2, 3)]
            )
            self.assertEqual(
                [row['step'] for row in result['task_acc_history']],
                ['event_0_CIL', 'event_2_CIL'],
            )
            self.assertEqual(
                set(result['task_acc_history'][-1]['per_task_accs']),
                {'task_0', 'task_1'},
            )

    def test_bwt_keeps_retained_comparisons_when_final_task_is_forgotten(self):
        tracker = MetricsTracker()
        tracker.record_task_accuracies(
            'event_10_CIL', {'task_0': 0.6}, 0.6
        )
        tracker.task_acc_matrix[-1].update({
            'deferred_diagonal': {'task_0': 1.0, 'task_1': 1.0},
            'deferred_final': True,
            'deferred_final_task': 'task_1',
        })
        self.assertEqual(tracker.compute_cl_metrics()['BWT'], -0.4)

    def test_no_test_loader_without_freeze_or_when_reload_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / 'adaptive_final.pt'
            torch.save({'schema_version': 1, 'trainer_state': {}}, final)
            dataset = _Dataset([])
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
                evaluate_deferred_cil_trajectory([], final, dataset, {}, args)
            self.assertEqual(dataset.test_accesses, [])

    def test_replaced_final_checkpoint_is_rejected_before_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            stage = save_deferred_cil_snapshot(
                _StateTrainer({'correct': {0: True}, 'fallback': 0}),
                _CLState(), args, 0, 0, {0: [0]},
            )
            final = run / 'adaptive_final.pt'
            torch.save({'schema_version': 1,
                        'protocol': {'device': 'cpu', 'num_parties': 0},
                        'trainer_state': {'bottoms': [], 'top_model': {
                'correct': {0: True}, 'fallback': 0,
            }}}, final)
            freeze = {
                'status': 'ADAPTIVE_STATE_FROZEN',
                'audit_spec': {'checkpoint': final.name},
                'checkpoint': {
                    'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
                },
                'snapshots': [_snapshot_record(stage, run)[1]],
            }
            atomic_write_new_json(run / 'ADAPTIVE_STATE_FROZEN.json', freeze)
            torch.save({'schema_version': 1,
                        'protocol': {'device': 'cpu', 'num_parties': 0},
                        'trainer_state': {'bottoms': [], 'top_model': {
                'correct': {0: False}, 'fallback': 0,
            }}}, final)
            dataset = _Dataset([])
            with mock.patch(
                'adaptive_consolidation_audit.audit_adaptive_checkpoint',
                return_value=copy.deepcopy(freeze),
            ), mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                return_value=_StateTrainer(),
            ):
                with self.assertRaises(ValueError):
                    evaluate_deferred_cil_trajectory(
                        [stage], final, dataset, {0: [0]}, args
                    )
            self.assertEqual(dataset.test_accesses, [])

    def test_stage_and_final_protocol_mismatch_precedes_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            stage = save_deferred_cil_snapshot(
                _StateTrainer({'correct': {0: True}, 'fallback': 0}),
                _CLState(), args, 0, 0, {0: [0]},
            )
            final = run / 'adaptive_final.pt'
            torch.save({
                'schema_version': 1,
                'protocol': {'device': 'cuda:0', 'num_parties': 0},
                'trainer_state': {'bottoms': [], 'top_model': {
                    'correct': {0: True}, 'fallback': 0,
                }},
            }, final)
            freeze = {
                'status': 'ADAPTIVE_STATE_FROZEN',
                'audit_spec': {'checkpoint': final.name},
                'checkpoint': {
                    'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
                },
                'snapshots': [_snapshot_record(stage, run)[1]],
            }
            atomic_write_new_json(run / 'ADAPTIVE_STATE_FROZEN.json', freeze)
            dataset = _Dataset([])
            with mock.patch(
                'adaptive_consolidation_audit.audit_adaptive_checkpoint',
                return_value=copy.deepcopy(freeze),
            ), mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                return_value=_StateTrainer(),
            ):
                with self.assertRaises(ValueError):
                    evaluate_deferred_cil_trajectory(
                        [stage], final, dataset, {0: [0]}, args
                    )
            self.assertEqual(dataset.test_accesses, [])

    def test_tampered_stage_snapshot_is_rejected_before_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(output_dir=tmp, device='cpu', num_parties=0)
            stage = save_deferred_cil_snapshot(
                _StateTrainer({'correct': {0: True}, 'fallback': 0}),
                _CLState(), args, 0, 0, {0: [0]}
            )
            final = run / 'adaptive_final.pt'
            torch.save({'schema_version': 1,
                        'protocol': {'device': 'cpu', 'num_parties': 0},
                        'trainer_state': {'bottoms': [], 'top_model': {
                'correct': {0: True}, 'fallback': 0,
            }}}, final)
            freeze = {
                'status': 'ADAPTIVE_STATE_FROZEN',
                'audit_spec': {'checkpoint': final.name},
                'checkpoint': {
                    'sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
                },
                'snapshots': [_snapshot_record(stage, run)[1]],
            }
            atomic_write_new_json(run / 'ADAPTIVE_STATE_FROZEN.json', freeze)
            payload = torch.load(stage, weights_only=False)
            payload['trainer_state']['top_model']['correct'][0] = False
            Path(stage).chmod(0o644)
            torch.save(payload, stage)
            Path(stage).chmod(0o444)
            dataset = _Dataset([])
            with mock.patch(
                'adaptive_consolidation_audit.audit_adaptive_checkpoint',
                return_value=copy.deepcopy(freeze),
            ), mock.patch(
                'adaptive_consolidation_audit._fresh_trainer',
                side_effect=lambda payload, _args: _StateTrainer(),
            ):
                with self.assertRaises((ValueError, RuntimeError)):
                    evaluate_deferred_cil_trajectory(
                        [stage], final, dataset, {0: [0]}, args
                    )
            self.assertEqual(dataset.test_accesses, [])


    def test_formal_snapshot_uses_separate_namespace_and_binds_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = SimpleNamespace(
                output_dir=tmp, save_task_checkpoints=0,
                formal_deferred_evaluation=True,
                seed=42, data='toy', cl_method='er', num_tasks=1,
                num_parties=0, head_consolidation_enabled=0,
                head_consolidation_mode='full_classifier',
            )
            trainer = _StrictStateTrainer()
            method = _CLState(7)
            tracker = MetricsTracker()
            comm = {'comm_rounds': 0, 'megabytes_transmitted': 0.0}
            tracker.record_timing('event_0_CIL', 0.25)
            tracker.record_comm('event_0_CIL', comm)
            tracker.record_step({
                'event_idx': 0, 'type': 'CIL', 'task_id': 0,
                'new_classes': [0, 1], 'evaluation_deferred': True,
                'train_time': 0.25, 'comm': comm,
            })
            _save_cil_checkpoint(
                trainer, method, args, 'event_0_CIL', 0, [0, 1],
                {0: [0, 1]}, tracker_state=tracker.to_dict(), force=True,
            )
            checkpoint = run / 'checkpoints' / 'event_0_CIL.pt'
            snapshot_trainer = _StrictStateTrainer()
            candidate_trainer = _StrictStateTrainer()
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state().clone()
            live_state_hash = trainer_state_sha256(trainer.get_state())

            stage = save_deferred_cil_snapshot(
                trainer, method, args, 0, 0, {0: [0, 1]},
                protocol_kind='formal',
            )

            self.assertEqual(Path(stage).parent.name, 'formal_snapshots')
            payload = torch.load(stage, map_location='cpu', weights_only=True)
            self.assertEqual(payload['kind'], 'formal_deferred_cil_snapshot')
            self.assertEqual(payload['protocol_kind'], 'formal')
            self.assertEqual(payload['cl_state'], {'value': 7})
            self.assertEqual(payload['tracker_state'], tracker.to_dict())
            self.assertTrue(_valid_tracker_checkpoint_state(
                payload['tracker_state'], payload['protocol']
            ))
            self.assertIn('rng_state', payload)
            self.assertEqual(payload['source_identity'], {
                'path': 'checkpoints/event_0_CIL.pt',
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'schema_version': 4,
                'step': 'event_0_CIL',
            })
            snapshot, record = _snapshot_record(
                stage, run, protocol_kind='formal'
            )
            self.assertEqual(
                snapshot['strict_reload_sha256'], payload['strict_reload_sha256']
            )
            self.assertEqual(snapshot['source_identity'], payload['source_identity'])
            self.assertEqual(record['protocol_kind'], 'formal')
            self.assertEqual(
                record['strict_reload_sha256'], payload['strict_reload_sha256']
            )
            self.assertEqual(
                _snapshot_manifest(run, protocol_kind='formal'), [record]
            )
            self.assertEqual(
                _strict_load_trainer_state(
                    snapshot_trainer, payload['trainer_state'], payload['protocol']
                ),
                trainer_state_sha256(payload['trainer_state']),
            )
            snapshot_method = _CLState(-1)
            snapshot_method.load_state(payload['cl_state'])
            self.assertEqual(snapshot_method.get_state(), payload['cl_state'])
            snapshot_tracker = MetricsTracker()
            snapshot_tracker.load_dict(payload['tracker_state'])
            self.assertEqual(snapshot_tracker.to_dict(), payload['tracker_state'])
            _validate_rng_checkpoint_state(payload['rng_state'])

            checkpoint_payload = torch.load(
                checkpoint, map_location='cpu', weights_only=True
            )

            class OneTask:
                def get_timeline(self):
                    return [{
                        'type': 'CIL', 'task_id': 0,
                        'new_classes': [0, 1],
                    }]

                def advance_task(self, _task_id):
                    pass

                def apply_unlearn(self, _classes):
                    pass

            decoded = _validate_resume_candidate_loadability(
                checkpoint_payload, args, candidate_trainer, _CLState(-1),
                OneTask(), mock.Mock(), MetricsTracker(),
            )
            self.assertEqual(decoded['tracker_state'], payload['tracker_state'])
            self.assertEqual(
                trainer_state_sha256(decoded['trainer_state']),
                trainer_state_sha256(payload['trainer_state']),
            )
            self.assertEqual(
                trainer_state_sha256(trainer.get_state()), live_state_hash
            )
            self.assertEqual(random.getstate(), python_rng)
            actual_numpy = np.random.get_state()
            self.assertEqual(actual_numpy[0], numpy_rng[0])
            self.assertTrue(np.array_equal(actual_numpy[1], numpy_rng[1]))
            self.assertEqual(actual_numpy[2:], numpy_rng[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    def test_formal_snapshot_rejects_snapshot_and_source_checkpoint_tampering(self):
        def fixture(root):
            args = SimpleNamespace(
                output_dir=str(root), save_task_checkpoints=0,
                formal_deferred_evaluation=True,
                seed=42, data='toy', cl_method='er', num_tasks=1,
                num_parties=0, head_consolidation_enabled=0,
                head_consolidation_mode='full_classifier',
            )
            trainer = _StateTrainer({'weight': torch.tensor([3.0])})
            method = _CLState(7)
            _save_cil_checkpoint(
                trainer, method, args, 'event_0_CIL', 0, [0, 1],
                {0: [0, 1]}, force=True,
            )
            stage = save_deferred_cil_snapshot(
                trainer, method, args, 0, 0, {0: [0, 1]},
                protocol_kind='formal',
            )
            return Path(stage), root / 'checkpoints' / 'event_0_CIL.pt'

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage, _ = fixture(run)
            payload = torch.load(stage, map_location='cpu', weights_only=True)
            payload['cl_state']['value'] = 99
            stage.chmod(0o644)
            torch.save(payload, stage)
            stage.chmod(0o444)
            with self.assertRaisesRegex(ValueError, 'strict reload'):
                _snapshot_record(stage, run, protocol_kind='formal')

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage, checkpoint = fixture(run)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            payload['task_id'] = 99
            checkpoint.chmod(0o644)
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, 'source checkpoint'):
                _snapshot_record(stage, run, protocol_kind='formal')

    def test_snapshot_top_metadata_is_required_typed_related_and_hash_bound(self):
        def fixture(root, protocol_kind):
            formal = protocol_kind == 'formal'
            args = SimpleNamespace(
                output_dir=str(root), save_task_checkpoints=0,
                formal_deferred_evaluation=formal,
                seed=42, data='toy', cl_method='er', num_tasks=1,
                num_parties=0, head_consolidation_enabled=0,
                head_consolidation_mode='full_classifier',
            )
            trainer = _StrictStateTrainer()
            method = _CLState(7)
            if formal:
                _save_cil_checkpoint(
                    trainer, method, args, 'event_0_CIL', 0, [0, 1],
                    {0: [0, 1]}, force=True,
                )
            return Path(save_deferred_cil_snapshot(
                trainer, method, args, 0, 0, {0: [0, 1]},
                protocol_kind=protocol_kind,
            ))

        mutations = {
            'garbage': lambda payload: payload.__setitem__('top_model', {
                'input_dim': '2', 'num_classes': 2, 'cosine': False,
            }),
            'missing': lambda payload: payload.pop('top_model'),
            'relationship': lambda payload: payload['top_model'].__setitem__(
                'input_dim', payload['top_model']['input_dim'] + 1
            ),
        }
        for protocol_kind in ('adaptive', 'formal'):
            for mutation, mutate in mutations.items():
                with self.subTest(
                        protocol_kind=protocol_kind, mutation=mutation), \
                        tempfile.TemporaryDirectory() as tmp:
                    run = Path(tmp)
                    stage = fixture(run, protocol_kind)
                    _snapshot_record(
                        stage, run, protocol_kind=protocol_kind
                    )
                    payload = torch.load(
                        stage, map_location='cpu', weights_only=True
                    )
                    mutate(payload)
                    stage.chmod(0o644)
                    torch.save(payload, stage)
                    stage.chmod(0o444)
                    with self.assertRaisesRegex(
                            ValueError, 'top|strict|identity'):
                        _snapshot_record(
                            stage, run, protocol_kind=protocol_kind
                        )

    def _formal_evaluation_fixture(self, root, method='er'):
        args = SimpleNamespace(
            output_dir=str(root), save_task_checkpoints=0,
            formal_deferred_evaluation=True, device='cpu',
            seed=42, data='toy', cl_method=method, num_tasks=2,
            num_parties=0, head_consolidation_enabled=0,
            head_consolidation_mode='full_classifier', bic_enabled=0,
            batch_size=2, num_classes=4, sanitize_cl_state=1,
            party_kd_enabled=0, party_kd_mode='uniform',
            party_proto_enabled=0, dep_tracking_enabled=0,
        )
        task_classes = {0: [0, 1], 1: [2, 3]}
        tracker = MetricsTracker()
        paths = []
        states = [
            {'correct': {0: True, 1: True}, 'fallback': 0},
            {'correct': {0: True, 1: False, 2: True, 3: True}, 'fallback': 0},
        ]
        for event_idx, task_id in enumerate((0, 1)):
            step = f'event_{event_idx}_CIL'
            comm = {'comm_rounds': event_idx, 'megabytes_transmitted': 0.0}
            tracker.record_timing(step, 0.25)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': event_idx, 'type': 'CIL', 'task_id': task_id,
                'new_classes': task_classes[task_id],
                'evaluation_deferred': True, 'train_time': 0.25,
                'comm': comm,
            })
            trainer = _FormalStateTrainer(states[event_idx])
            cl_method = get_cl_method(method, trainer, args)
            if method == 'er':
                cl_method.buffer.add_batch(
                    torch.tensor([[float(event_idx)]]),
                    torch.tensor([event_idx]),
                )
            if method == 'proto_evolve':
                cl_method.current_task_id = event_idx
            method_state = cl_method.get_state()
            method_probe = get_cl_method(method, trainer, args)
            method_probe.load_state(copy.deepcopy(method_state))
            self.assertTrue(runner._checkpoint_values_equal(
                method_probe.get_state(), method_state
            ))
            seen = {
                key: task_classes[key] for key in range(task_id + 1)
            }
            _save_cil_checkpoint(
                trainer, cl_method, args, step, task_id,
                task_classes[task_id], seen,
                tracker_state=tracker.to_dict(), force=True,
            )
            paths.append(Path(save_deferred_cil_snapshot(
                trainer, cl_method, args, event_idx, task_id,
                seen, protocol_kind='formal',
            )))
        final_checkpoint = root / 'checkpoints' / 'event_1_CIL.pt'
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        cache = cache_formal_batches([
            (torch.arange(4, dtype=torch.float32).view(4, 1), labels[:4]),
            (torch.arange(4, 8, dtype=torch.float32).view(4, 1), labels[4:]),
        ])
        return args, paths, final_checkpoint, task_classes, cache

    def _formal_bic_evaluation_fixture(self, root):
        args = SimpleNamespace(
            output_dir=str(root), save_task_checkpoints=0,
            formal_deferred_evaluation=True, device='cpu',
            seed=42, data='cifar100', cl_method='finetune', num_tasks=10,
            num_parties=0, head_consolidation_enabled=0,
            head_consolidation_mode='full_classifier', bic_enabled=1,
            bic_fit_mode='joint_each_stage', bic_lr=0.05, bic_steps=2,
            bic_per_class=2, batch_size=8, num_classes=20,
            lambda_validation_enabled=1, lambda_validation_per_class=1,
            lambda_validation_split_seed=7, sanitize_cl_state=1,
            party_kd_enabled=0, party_kd_mode='uniform',
            party_proto_enabled=0, dep_tracking_enabled=0,
        )
        task_classes = {
            task_id: [2 * task_id, 2 * task_id + 1]
            for task_id in range(10)
        }
        tracker = MetricsTracker()
        paths = []
        for task_id in range(10):
            step = f'event_{task_id}_CIL'
            comm = {'comm_rounds': task_id, 'megabytes_transmitted': 0.0}
            tracker.record_timing(step, 0.25)
            tracker.record_comm(step, comm)
            tracker.record_step({
                'event_idx': task_id, 'type': 'CIL', 'task_id': task_id,
                'new_classes': task_classes[task_id],
                'evaluation_deferred': True, 'train_time': 0.25,
                'comm': comm,
            })
            trainer = _FormalBiCTrainer({
                'correct': {
                    class_id: True for class_id in range(2 * (task_id + 1))
                },
                'fallback': 0,
            })
            method = get_cl_method('finetune', trainer, args)
            seen = {
                key: task_classes[key] for key in range(task_id + 1)
            }
            _save_cil_checkpoint(
                trainer, method, args, step, task_id, task_classes[task_id],
                seen, tracker_state=tracker.to_dict(), force=True,
            )
            paths.append(Path(save_deferred_cil_snapshot(
                trainer, method, args, task_id, task_id, seen,
                protocol_kind='formal',
            )))
        source = root / 'checkpoints' / 'event_9_CIL.pt'
        final_checkpoint = root / 'checkpoints' / 'formal_final.pt'
        _atomic_torch_save(
            torch.load(source, map_location='cpu', weights_only=True),
            final_checkpoint,
        )
        labels = torch.arange(20, dtype=torch.long).repeat(2)
        rows = labels.to(torch.float32).view(-1, 1)
        calibration = cache_formal_batches((
            (rows[:20], labels[:20]), (rows[20:], labels[20:]),
        ))
        test = cache_formal_batches((
            (rows[:10], labels[:10]), (rows[10:30], labels[10:30]),
            (rows[30:], labels[30:]),
        ))
        audit = {
            'passed': True, 'manifest_sha256': 'a' * 64,
            'per_class': 2, 'calibration_count': 40,
            'training_count': 9940, 'overlap_count': 0,
            'test_used_for_fit': False,
        }
        ordered_indices = list(range(20))
        validation_sha = hashlib.sha256(json.dumps(
            ordered_indices, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        validation_manifest = {
            'dataset': 'cifar100-train', 'seed': 7, 'per_class': 1,
            'by_class': {
                str(class_id): [class_id] for class_id in range(20)
            },
            'ordered_indices': ordered_indices,
            'sha256': validation_sha,
        }
        selection_audit = {
            'passed': True,
            'calibration_manifest_sha256': 'a' * 64,
            'validation_manifest_sha256': validation_sha,
            'calibration_per_class': 2,
            'validation_per_class': 1,
            'training_count': 9940,
            'calibration_count': 40,
            'validation_count': 20,
            'training_calibration_overlap_count': 0,
            'training_validation_overlap_count': 0,
            'calibration_validation_overlap_count': 0,
            'evaluation_source': 'cifar100-train-validation',
            'test_used_for_selection': False,
        }
        authoritative_manifest = {
            'logical_path': 'validation/validation_manifest.json',
            'dataset': 'cifar100-train', 'seed': 7, 'per_class': 1,
            'sha256': validation_sha,
        }
        return (args, paths, final_checkpoint, task_classes,
                calibration, test, audit, validation_manifest,
                selection_audit, authoritative_manifest)

    def _run_formal_evaluation(self, fixture):
        args, paths, final_checkpoint, task_classes, cache = fixture
        prepare_formal_deferred_evaluation(
            args=args, snapshot_paths=paths,
            final_checkpoint=final_checkpoint, task_classes=task_classes,
            output_dir=args.output_dir,
        )
        created = []

        def fresh(_payload, _args):
            trainer = _FormalStateTrainer()
            created.append(trainer)
            return trainer

        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        cache_before = copy.deepcopy(cache)
        with mock.patch(
                'adaptive_consolidation_audit._fresh_trainer', side_effect=fresh):
            result = evaluate_formal_deferred_trajectory(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint,
                task_classes=task_classes, cached_test_batches=cache,
                output_dir=args.output_dir,
            )
        return result, created, cache_before, python_rng, numpy_rng, torch_rng

    @mock.patch(
        'adaptive_consolidation_audit._formal_source_provenance',
        return_value={
            'schema_version': 1,
            'source_commit': 'a' * 40,
            'source_sha256': {
                name: 'b' * 64
                for name in consolidation_audit.FORMAL_SOURCE_FILES
            },
        },
    )
    def test_formal_recompute_snapshot_bytes_are_loaded_stage_by_stage(
            self, _provenance):
        with tempfile.TemporaryDirectory() as tmp:
            args, paths, final_path, tasks, cache = self._formal_evaluation_fixture(
                Path(tmp)
            )
            final = torch.load(final_path, map_location='cpu', weights_only=True)
            decoded = [torch.load(path, map_location='cpu', weights_only=True)
                       for path in paths]
            raw = [path.read_bytes() for path in paths]
            events = []
            original_load = consolidation_audit._restricted_torch_load
            original_fresh = consolidation_audit._fresh_formal_state

            def load(content):
                payload = original_load(content)
                events.append(('load', payload['event_idx']))
                return payload

            def fresh(payload, isolated_args):
                events.append(('state', payload['event_idx']))
                return original_fresh(payload, isolated_args)

            inputs = dict(args=args, snapshot_paths=paths,
                          final_checkpoint=final_path, task_classes=tasks,
                          cached_test_batches=cache, output_dir=tmp)
            with mock.patch('adaptive_consolidation_audit._fresh_trainer',
                            side_effect=lambda _payload, _args: _FormalStateTrainer()):
                baseline = evaluate_formal_deferred_trajectory(
                    **inputs, recompute_payloads={'snapshots': decoded,
                                                  'final': final})
                with mock.patch('adaptive_consolidation_audit._restricted_torch_load',
                                side_effect=load), mock.patch(
                        'adaptive_consolidation_audit._fresh_formal_state',
                        side_effect=fresh):
                    streamed = evaluate_formal_deferred_trajectory(
                        **inputs, recompute_payloads={'snapshots': raw,
                                                      'final': final})
                self.assertTrue(runner._checkpoint_values_equal(baseline, streamed))
                self.assertEqual(events[:4], [('load', 0), ('state', 0),
                                              ('load', 1), ('state', 1)])
                with self.assertRaisesRegex(
                        ValueError, 'restricted adaptive checkpoint load failed'):
                    evaluate_formal_deferred_trajectory(
                        **inputs, recompute_payloads={
                            'snapshots': [b'invalid checkpoint', raw[1]],
                            'final': final,
                        })

    def test_formal_each_state_reuses_only_its_exact_union_logits(self):
        from formal_cifar100_metrics import calibration_history, compute_formal_metrics
        from test_formal_cifar100_metrics import _legacy_formal_readouts

        original_fresh = consolidation_audit._fresh_formal_state
        original_readout = runner._evaluate_cil_readouts_cached
        original_fit = runner._fit_and_evaluate_final_bic_cached
        source_root = Path(__file__).resolve().parent
        source_provenance = {
            'schema_version': 1,
            'source_commit': subprocess.check_output([
                'git', '-C', str(source_root), 'rev-parse', 'HEAD',
            ], text=True).strip(),
            'source_sha256': {
                logical: hashlib.sha256(subprocess.check_output([
                    'git', '-C', str(source_root), 'show', f'HEAD:{logical}',
                ])).hexdigest()
                for logical in consolidation_audit.FORMAL_SOURCE_FILES
            },
        }

        class RecordedTrainer(_FormalBiCTrainer):
            def __init__(self):
                super().__init__()
                self.forward_rows = []
                self.collected = []

            def values(self, batches, kind):
                self.forward_rows.extend(
                    (kind, tuple(inputs[:, 0].tolist())) for inputs, _ in batches
                )
                logits, labels = super().collect_logits(batches)
                logits[:, 0] += self.source_index * 0.7
                return logits, labels

            def evaluate(self, batches):
                logits, labels = self.values(batches, 'evaluate')
                return (int(logits.argmax(1).eq(labels).sum()) / labels.numel(),
                        torch.softmax(logits, 1), labels)

            def collect_logits(self, batches):
                values = self.values(batches, 'collect')
                self.collected.append(tuple(value.clone() for value in values))
                return values

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                'adaptive_consolidation_audit._formal_source_provenance',
                side_effect=lambda _root=None: copy.deepcopy(source_provenance)):
            fixture = self._formal_bic_evaluation_fixture(Path(tmp))
            (args, paths, final_checkpoint, tasks, calibration, cached,
             audit, manifest, selection, authoritative) = fixture
            payloads = {
                'snapshots': [torch.load(path, map_location='cpu', weights_only=True)
                              for path in paths],
                'final': torch.load(final_checkpoint, map_location='cpu', weights_only=True),
            }
            labels = torch.cat([value for _, value in cached])
            rows = torch.arange(labels.numel()).float().view(-1, 1)
            cached = cache_formal_batches(((rows[:10], labels[:10]),
                                          (rows[10:30], labels[10:30]),
                                          (rows[30:], labels[30:])))
            calibration = cache_formal_batches(((rows + 100, labels),))
            identities = tuple(consolidation_audit._formal_cache_identity(value)
                               for value in (cached, calibration))
            rng_before = runner._capture_rng_state()
            fitted_state_indices = set(range(9)) | {10}
            runs = []
            for reuse in (False, True):
                states, readouts, bundles, fits = [], {}, {}, []
                reference_sentinels = {}

                def fresh(payload, isolated_args):
                    trainer, method, tracker = original_fresh(payload, isolated_args)
                    trainer.source_index = len(states)
                    states.append(trainer)
                    return trainer, method, tracker

                def readout(method, trainer, *values, **kwargs):
                    index = trainer.source_index
                    if reuse:
                        self.assertEqual(
                            kwargs.get('collect_union_logits', False),
                            index in fitted_state_indices,
                        )
                        result = original_readout(method, trainer, *values, **kwargs)
                        if index in fitted_state_indices:
                            self.assertIn('formal_union_logits', result)
                            bundles[index] = result['formal_union_logits']
                        else:
                            self.assertNotIn('formal_union_logits', result)
                    else:
                        result = _legacy_formal_readouts(method, trainer, *values)
                        if index in fitted_state_indices:
                            sentinel = object()
                            reference_sentinels[index] = sentinel
                            result['formal_union_logits'] = sentinel
                        else:
                            self.assertNotIn('formal_union_logits', result)
                    readouts[index] = {key: value for key, value in result.items()
                                       if key != 'formal_union_logits'}
                    return result

                def fit(*values, **kwargs):
                    trainer = values[0]
                    index = trainer.source_index
                    if reuse:
                        self.assertIs(kwargs.get('precomputed_test'), bundles[index])
                        record = original_fit(*values, **kwargs)
                        fit_values = trainer.collected[-1]
                        test_values = (bundles[index]['logits'], bundles[index]['labels'])
                    else:
                        self.assertIs(
                            kwargs.pop('precomputed_test', None),
                            reference_sentinels[index],
                        )
                        record = original_fit(*values, **kwargs)
                        fit_values, test_values = trainer.collected[-2:]
                    fits.append((index, fit_values, test_values))
                    return record

                with mock.patch.dict(
                        'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                        {'cifar100': authoritative}), mock.patch(
                        'adaptive_consolidation_audit._fresh_trainer',
                        side_effect=lambda _payload, _args: RecordedTrainer()), mock.patch(
                        'adaptive_consolidation_audit._fresh_formal_state',
                        side_effect=fresh), mock.patch(
                        'runner._evaluate_cil_readouts_cached', side_effect=readout), mock.patch(
                        'runner._fit_and_evaluate_final_bic_cached', side_effect=fit):
                    outcome = evaluate_formal_deferred_trajectory(
                        args=args, snapshot_paths=paths, final_checkpoint=final_checkpoint,
                        task_classes=tasks, cached_test_batches=cached,
                        cached_calibration_batches=calibration, calibration_audit=audit,
                        validation_manifest=manifest, selection_audit=selection,
                        output_dir=tmp, recompute_payloads=payloads,
                    )
                self.assertTrue(runner._checkpoint_values_equal(
                    runner._capture_rng_state(), rng_before))
                self.assertEqual(tuple(consolidation_audit._formal_cache_identity(value)
                                       for value in (cached, calibration)), identities)
                self.assertEqual(len(states), 11)
                self.assertIsNot(states[9], states[10])
                self.assertEqual([value[0] for value in fits], list(range(9)) + [10])
                if not reuse:
                    self.assertEqual(set(reference_sentinels), fitted_state_indices)
                runs.append((outcome, states, readouts, bundles, fits))

            reference, current = runs
            self.assertEqual(json.dumps(current[0], sort_keys=True),
                             json.dumps(reference[0], sort_keys=True))
            self.assertEqual(current[2], reference[2])
            self.assertEqual(set(current[3]), set(range(9)) | {10})
            self.assertEqual(len({id(value) for value in current[3].values()}), 10)
            self.assertNotIn('formal_union_logits', json.dumps(current[0], sort_keys=True))
            self.assertEqual(
                compute_formal_metrics(calibration_history(current[0]['bic_history']), expected_tasks=10),
                compute_formal_metrics(calibration_history(reference[0]['bic_history']), expected_tasks=10),
            )
            removed_rows = 0
            removed_calls = 0
            for before, after in zip(reference[4], current[4]):
                self.assertEqual(before[0], after[0])
                for old_values, new_values in zip(before[1:], after[1:]):
                    for old_tensor, new_tensor in zip(old_values, new_values):
                        self.assertTrue(torch.equal(old_tensor, new_tensor))
                index = before[0]
                old_rows, new_rows = (run[1][index].forward_rows for run in runs)
                union_batches = [rows for kind, rows in new_rows
                                 if kind == 'collect' and rows[0] < 100]
                removed_rows += sum(len(rows) for rows in union_batches)
                removed_calls += len(union_batches)
                self.assertEqual(sum(len(rows) for _, rows in old_rows)
                                 - sum(len(rows) for _, rows in new_rows), len(before[2][1]))
                # Original per-task batch forwards precede the same union collection.
                per_task = len(new_rows) - len(union_batches) - 1
                self.assertEqual(old_rows[:per_task], new_rows[:per_task])
            self.assertEqual(reference[1][9].forward_rows, current[1][9].forward_rows)
            self.assertEqual(removed_rows, 220)
            self.assertEqual(removed_calls, 25)

    def test_formal_fedprotip_bic_uses_legacy_test_logits_fallback(self):
        original_fresh = consolidation_audit._fresh_formal_state
        original_readout = runner._evaluate_cil_readouts_cached
        original_fit = runner._fit_and_evaluate_final_bic_cached
        source_root = Path(__file__).resolve().parent
        source_provenance = {
            'schema_version': 1,
            'source_commit': subprocess.check_output([
                'git', '-C', str(source_root), 'rev-parse', 'HEAD',
            ], text=True).strip(),
            'source_sha256': {
                logical: hashlib.sha256(subprocess.check_output([
                    'git', '-C', str(source_root), 'show', f'HEAD:{logical}',
                ])).hexdigest()
                for logical in consolidation_audit.FORMAL_SOURCE_FILES
            },
        }

        class FedProTIPLikeMethod:
            def evaluate_class_il_readouts_cached(
                    self, _cached_batches, task_classes):
                keys = [f'task_{task_id}' for task_id in task_classes]
                return {
                    'class_il_pred_task': {key: 0.5 for key in keys},
                    'class_il_global': {key: 0.25 for key in keys},
                    'task_il_oracle': {key: 0.75 for key in keys},
                    'task_prediction': {key: 1.0 for key in keys},
                    'class_il_pred_task_counts': {
                        key: {'correct': 2, 'total': 4} for key in keys
                    },
                }

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                'adaptive_consolidation_audit._formal_source_provenance',
                side_effect=lambda _root=None: copy.deepcopy(source_provenance)):
            fixture = self._formal_bic_evaluation_fixture(Path(tmp))
            (args, paths, final_checkpoint, tasks, calibration, cached,
             audit, manifest, selection, authoritative) = fixture
            payloads = {
                'snapshots': [torch.load(path, map_location='cpu', weights_only=True)
                              for path in paths],
                'final': torch.load(
                    final_checkpoint, map_location='cpu', weights_only=True
                ),
            }
            requested_union_cache = []
            fit_precomputed_test = []

            def fresh(payload, isolated_args):
                trainer, _method, tracker = original_fresh(
                    payload, isolated_args
                )
                return trainer, FedProTIPLikeMethod(), tracker

            def readout(*values, **kwargs):
                requested_union_cache.append('collect_union_logits' in kwargs)
                result = original_readout(*values, **kwargs)
                self.assertNotIn('formal_union_logits', result)
                return result

            def fit(*values, **kwargs):
                fit_precomputed_test.append('precomputed_test' in kwargs)
                return original_fit(*values, **kwargs)

            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative}), mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalBiCTrainer()), mock.patch(
                    'adaptive_consolidation_audit._fresh_formal_state',
                    side_effect=fresh), mock.patch(
                    'runner._evaluate_cil_readouts_cached',
                    side_effect=readout), mock.patch(
                    'runner._fit_and_evaluate_final_bic_cached',
                    side_effect=fit):
                outcome = evaluate_formal_deferred_trajectory(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint, task_classes=tasks,
                    cached_test_batches=cached,
                    cached_calibration_batches=calibration,
                    calibration_audit=audit,
                    validation_manifest=manifest,
                    selection_audit=selection, output_dir=tmp,
                    recompute_payloads=payloads,
                )

            self.assertEqual(requested_union_cache, [False] * 11)
            self.assertEqual(fit_precomputed_test, [False] * 10)
            self.assertEqual(len(outcome['bic_history']), 10)
            final = outcome['tracker_state']['task_acc_history'][-1]
            expected_keys = [f'task_{task_id}' for task_id in tasks]
            self.assertEqual(
                final['per_task_accs'], {key: 0.5 for key in expected_keys}
            )
            self.assertEqual(final['overall_acc'], 0.5)
            self.assertNotIn('per_task_accs_debiased', final)
            self.assertEqual(
                final['per_task_accs_taskil'],
                {key: 0.75 for key in expected_keys},
            )
            self.assertEqual(final['companion_readouts'], {
                'class_il_global': {key: 0.25 for key in expected_keys},
                'task_prediction': {key: 1.0 for key in expected_keys},
            })

    def test_formal_bic_reconstructs_ten_stages_and_publishes_exact_state(self):
        from bic_calibration import TaskAffineCalibrator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._formal_bic_evaluation_fixture(root)
            (args, paths, final_checkpoint, task_classes,
             calibration, test, audit, validation_manifest,
             selection_audit, authoritative_manifest) = fixture
            prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint, task_classes=task_classes,
                output_dir=tmp,
            )
            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative_manifest}), mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalBiCTrainer()):
                outcome = evaluate_formal_deferred_trajectory(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes,
                    cached_test_batches=test,
                    cached_calibration_batches=calibration,
                    calibration_audit=audit,
                    validation_manifest=validation_manifest,
                    selection_audit=selection_audit,
                    output_dir=tmp,
                )

            self.assertEqual(set(outcome), {
                'tracker_state', 'bic_history', 'bic_state',
                'calibration_audit', 'bic_fit_corpus',
            })
            history = outcome['bic_history']
            self.assertEqual(len(history), 10)
            self.assertEqual(
                [(record['step'], record['task_id']) for record in history],
                [(f'event_{task}_CIL', task) for task in range(10)],
            )
            self.assertTrue(all(
                record['calibration_audit'] == audit
                and record['fit_corpus'] == outcome['bic_fit_corpus']
                for record in history
            ))
            self.assertEqual(outcome['bic_fit_corpus']['split'], 'calibration')
            self.assertEqual(
                outcome['bic_fit_corpus']['manifest_sha256'], 'a' * 64
            )
            self.assertEqual(
                [set(record['parameters']) for record in history],
                [{str(value) for value in range(task + 1)}
                 for task in range(10)],
            )
            restored = TaskAffineCalibrator()
            restored.load_state_dict(outcome['bic_state'])
            self.assertEqual(
                {str(task): restored.parameters_for(task) for task in range(10)},
                history[-1]['parameters'],
            )
            complete = json.loads(
                (root / 'FORMAL_EVALUATION_COMPLETE.json').read_text()
            )
            self.assertEqual(complete['bic']['history'], history)
            self.assertEqual(complete['bic']['state'], outcome['bic_state'])
            self.assertEqual(
                complete['bic']['fit_corpus'], outcome['bic_fit_corpus']
            )
            self.assertIn('calibration_cache_identity', complete)
            self.assertNotIn('cached_calibration_batches', repr(complete))

            final = {
                **outcome['tracker_state'],
                'bic_history': history,
                'bic_final': history[-1],
                'calibration_audit': audit,
                'bic_fit_corpus': outcome['bic_fit_corpus'],
                'selection_audit': selection_audit,
                'config': {'cl_method': 'finetune'},
            }
            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative_manifest}), mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalBiCTrainer()):
                _publish_formal_deferred_result(
                    args=args, final_checkpoint=final_checkpoint,
                    tracker_state=outcome['tracker_state'], final_result=final,
                    bic_state=outcome['bic_state'], bic_history=history,
                )
            published = torch.load(
                final_checkpoint, map_location='cpu', weights_only=True
            )
            self.assertEqual(published['schema_version'], 4)
            self.assertEqual(published['tracker_state'], outcome['tracker_state'])
            self.assertEqual(published['bic_history'], final['bic_history'])
            self.assertEqual(published['bic_state'], outcome['bic_state'])
            self.assertEqual(json.loads(
                (root / 'results.json').read_text()), final)

    def test_formal_bic_cache_tamper_and_crash_fail_closed(self):
        for case in ('tamper', 'crash'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = self._formal_bic_evaluation_fixture(root)
                (args, paths, final_checkpoint, task_classes,
                 calibration, test, audit, validation_manifest,
                 selection_audit, authoritative_manifest) = fixture
                prepare_formal_deferred_evaluation(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes, output_dir=tmp,
                )
                original_write = atomic_write_new_json
                original_fit = getattr(
                    runner, '_fit_and_evaluate_final_bic_cached', None
                )
                self.assertTrue(callable(original_fit))

                def mutate(*call_args, **call_kwargs):
                    record = original_fit(*call_args, **call_kwargs)
                    calibration[0][0][0, 0] += 1.0
                    return record

                def crash(path, payload):
                    if Path(path).name == 'FORMAL_EVALUATION_COMPLETE.json':
                        raise RuntimeError('simulated BiC complete crash')
                    return original_write(path, payload)

                patches = [mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalBiCTrainer(),
                )]
                if case == 'tamper':
                    patches.append(mock.patch(
                        'runner._fit_and_evaluate_final_bic_cached',
                        side_effect=mutate,
                    ))
                else:
                    patches.append(mock.patch(
                        'adaptive_consolidation_audit.atomic_write_new_json',
                        side_effect=crash,
                    ))
                with mock.patch.dict(
                        'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                        {'cifar100': authoritative_manifest}), \
                        patches[0], patches[1]:
                    with self.assertRaisesRegex(
                            RuntimeError, 'mutated|complete crash'):
                        evaluate_formal_deferred_trajectory(
                            args=args, snapshot_paths=paths,
                            final_checkpoint=final_checkpoint,
                            task_classes=task_classes,
                            cached_test_batches=test,
                            cached_calibration_batches=calibration,
                            calibration_audit=audit,
                            validation_manifest=validation_manifest,
                            selection_audit=selection_audit,
                            output_dir=tmp,
                        )
                self.assertTrue(
                    (root / 'FORMAL_EVALUATION_CONSUMING.json').is_file()
                )
                self.assertFalse(
                    (root / 'FORMAL_EVALUATION_COMPLETE.json').exists()
                )
                with self.assertRaisesRegex(RuntimeError, 'consuming|fail.closed'):
                    prepare_formal_deferred_evaluation(
                        args=args, snapshot_paths=paths,
                        final_checkpoint=final_checkpoint,
                        task_classes=task_classes, output_dir=tmp,
                    )

    def test_formal_bic_final_stage_uses_installed_final_trainer(self):
        import adaptive_consolidation_audit as audit_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._formal_bic_evaluation_fixture(root)
            (args, paths, final_checkpoint, task_classes,
             calibration, test, audit, validation_manifest,
             selection_audit, authoritative_manifest) = fixture
            prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint,
                task_classes=task_classes, output_dir=tmp,
            )
            original_fresh = audit_module._fresh_formal_state
            original_fit = runner._fit_and_evaluate_final_bic_cached
            fresh_count = 0
            fit_sources = []
            fitted_states = []

            def fresh(payload, isolated_args):
                nonlocal fresh_count
                trainer, method, tracker = original_fresh(
                    payload, isolated_args
                )
                trainer.state['formal_source'] = (
                    'installed_final' if fresh_count == 10
                    else f'stage_{fresh_count}'
                )
                fresh_count += 1
                return trainer, method, tracker

            def fit(*call_args, **call_kwargs):
                trainer, calibrator = call_args[0], call_args[3]
                fit_sources.append(trainer.state['formal_source'])
                record = original_fit(*call_args, **call_kwargs)
                fitted_states.append(copy.deepcopy(calibrator.state_dict()))
                return record

            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative_manifest}), mock.patch(
                        'adaptive_consolidation_audit._fresh_trainer',
                        side_effect=lambda _payload, _args:
                        _FormalBiCTrainer()), mock.patch(
                            'adaptive_consolidation_audit._fresh_formal_state',
                            side_effect=fresh), mock.patch(
                                'runner._fit_and_evaluate_final_bic_cached',
                                side_effect=fit):
                outcome = evaluate_formal_deferred_trajectory(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes,
                    cached_test_batches=test,
                    cached_calibration_batches=calibration,
                    calibration_audit=audit,
                    validation_manifest=validation_manifest,
                    selection_audit=selection_audit,
                    output_dir=tmp,
                )
            self.assertEqual(
                fit_sources,
                [f'stage_{task_id}' for task_id in range(9)]
                + ['installed_final'],
            )
            self.assertEqual(len(outcome['bic_history']), 10)
            self.assertEqual(outcome['bic_state'], fitted_states[-1])

    def test_formal_bic_complete_reuses_exact_producer_schema_validator(self):
        mutations = {
            'missing_fit': lambda bic: bic['history'][0].pop('fit'),
            'extra_record_key': lambda bic: bic['history'][0].__setitem__(
                'unexpected', True
            ),
            'privacy_false': lambda bic: bic['history'][0][
                'privacy_audit'
            ].__setitem__('passed', False),
            'paired_value': lambda bic: bic['history'][0]['paired']['raw']
                .__setitem__('overall_accuracy', 0.123),
            'parameter_type': lambda bic: bic['history'][-1]['parameters']['0']
                .__setitem__('alpha', 1),
            'manifest_hash': lambda bic: bic['validation_manifest']
                .__setitem__('sha256', 'f' * 64),
            'selection_failed': lambda bic: bic['selection_audit']
                .__setitem__('passed', False),
            'selection_validation_hash': lambda bic: bic['selection_audit']
                .__setitem__('validation_manifest_sha256', 'f' * 64),
            'selection_calibration_hash': lambda bic: bic['selection_audit']
                .__setitem__('calibration_manifest_sha256', 'f' * 64),
            'calibration_manifest_hash': lambda bic: bic['calibration_audit']
                .__setitem__('manifest_sha256', 'f' * 64),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._formal_bic_evaluation_fixture(root)
            (args, paths, final_checkpoint, task_classes,
             calibration, test, audit, validation_manifest,
             selection_audit, authoritative_manifest) = fixture
            prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint,
                task_classes=task_classes, output_dir=tmp,
            )
            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative_manifest}), mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalBiCTrainer()):
                evaluate_formal_deferred_trajectory(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes,
                    cached_test_batches=test,
                    cached_calibration_batches=calibration,
                    calibration_audit=audit,
                    validation_manifest=validation_manifest,
                    selection_audit=selection_audit,
                    output_dir=tmp,
                )
            complete = json.loads(
                (root / 'FORMAL_EVALUATION_COMPLETE.json').read_text()
            )
            with mock.patch.dict(
                    'three_dataset_formal_audit._AUTHORITATIVE_MANIFEST',
                    {'cifar100': authoritative_manifest}):
                for name, mutate in mutations.items():
                    with self.subTest(name=name):
                        forged = copy.deepcopy(complete['bic'])
                        mutate(forged)
                        with self.assertRaises(ValueError):
                            _validate_formal_bic_bundle(
                                forged, complete['identity']
                            )

    def test_formal_cached_stage_evaluation_is_generic_strict_and_immutable(self):
        for method in ('finetune', 'proto_evolve', 'er', 'ewc'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                fixture = self._formal_evaluation_fixture(Path(tmp), method)
                live = _StateTrainer({'weight': torch.tensor([9.0])})
                live_hash = trainer_state_sha256(live.get_state())
                result, created, cache_before, python_rng, numpy_rng, torch_rng = \
                    self._run_formal_evaluation(fixture)

                rows = result['task_acc_history']
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['per_task_accs'], {'task_0': 1.0})
                self.assertEqual(rows[-1]['per_task_accs'], {
                    'task_0': 0.5, 'task_1': 1.0,
                })
                self.assertEqual(rows[-1]['per_task_accs_taskil'], {
                    'task_0': 0.5, 'task_1': 1.0,
                })
                self.assertEqual(rows[-1]['deferred_diagonal'], {
                    'task_0': 1.0, 'task_1': 1.0,
                })
                self.assertEqual(result['cl_metrics']['AA_final'], 0.75)
                self.assertEqual(result['cl_metrics']['BWT'], -0.5)
                self.assertEqual(len(created), 3)
                self.assertEqual(
                    [trainer.state for trainer in created],
                    [
                        {'correct': {0: True, 1: True}, 'fallback': 0},
                        {'correct': {
                            0: True, 1: False, 2: True, 3: True,
                        }, 'fallback': 0},
                        {'correct': {
                            0: True, 1: False, 2: True, 3: True,
                        }, 'fallback': 0},
                    ],
                )
                self.assertEqual(trainer_state_sha256(live.get_state()), live_hash)
                for before, after in zip(cache_before, fixture[-1]):
                    self.assertTrue(torch.equal(before[0], after[0]))
                    self.assertTrue(torch.equal(before[1], after[1]))
                self.assertEqual(random.getstate(), python_rng)
                actual_numpy = np.random.get_state()
                self.assertEqual(actual_numpy[0], numpy_rng[0])
                self.assertTrue(np.array_equal(actual_numpy[1], numpy_rng[1]))
                self.assertEqual(actual_numpy[2:], numpy_rng[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                artifact = json.loads(
                    (Path(tmp) / 'FORMAL_EVALUATION_COMPLETE.json').read_text()
                )
                self.assertNotIn('cached_test_batches', repr(artifact))
                self.assertIn('cache_identity', artifact)

    def test_formal_pending_consuming_and_complete_restart_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._formal_evaluation_fixture(Path(tmp))
            args, paths, final_checkpoint, task_classes, _cache = fixture
            prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint, task_classes=task_classes,
                output_dir=tmp,
            )
            with self.assertRaisesRegex(RuntimeError, 'pending|fail.closed'):
                prepare_formal_deferred_evaluation(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes, output_dir=tmp,
                )

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._formal_evaluation_fixture(Path(tmp))
            args, paths, final_checkpoint, task_classes, cache = fixture
            prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint, task_classes=task_classes,
                output_dir=tmp,
            )
            original = atomic_write_new_json

            def crash(path, payload):
                if Path(path).name == 'FORMAL_EVALUATION_COMPLETE.json':
                    raise RuntimeError('simulated artifact crash')
                return original(path, payload)

            with mock.patch(
                    'adaptive_consolidation_audit.atomic_write_new_json',
                    side_effect=crash), mock.patch(
                        'adaptive_consolidation_audit._fresh_trainer',
                        side_effect=lambda _payload, _args: _FormalStateTrainer()):
                with self.assertRaisesRegex(RuntimeError, 'artifact crash'):
                    evaluate_formal_deferred_trajectory(
                        args=args, snapshot_paths=paths,
                        final_checkpoint=final_checkpoint,
                        task_classes=task_classes,
                        cached_test_batches=cache, output_dir=tmp,
                    )
            self.assertTrue(
                (Path(tmp) / 'FORMAL_EVALUATION_CONSUMING.json').is_file()
            )
            with self.assertRaisesRegex(RuntimeError, 'consuming|fail.closed'):
                prepare_formal_deferred_evaluation(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes, output_dir=tmp,
                )

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._formal_evaluation_fixture(Path(tmp))
            result, *_ = self._run_formal_evaluation(fixture)
            args, paths, final_checkpoint, task_classes, _cache = fixture
            reused = prepare_formal_deferred_evaluation(
                args=args, snapshot_paths=paths,
                final_checkpoint=final_checkpoint,
                task_classes=task_classes, output_dir=tmp,
            )
            self.assertEqual(reused['status'], 'complete')
            self.assertEqual(reused['result'], result)

    def test_formal_complete_seal_rejects_forged_cache_and_evaluation_without_reopen(self):
        for forgery in ('cache', 'evaluation'):
            with self.subTest(forgery=forgery), \
                    tempfile.TemporaryDirectory() as tmp:
                fixture = self._formal_evaluation_fixture(Path(tmp))
                self._run_formal_evaluation(fixture)
                complete_path = Path(tmp) / 'FORMAL_EVALUATION_COMPLETE.json'
                complete = json.loads(complete_path.read_text())
                if forgery == 'cache':
                    complete['cache_identity']['sample_count'] += 1
                else:
                    tracker = MetricsTracker()
                    tracker.load_dict(complete['evaluation'])
                    tracker.task_acc_matrix[-1]['per_task_accs']['task_0'] = 0.25
                    complete['evaluation'] = tracker.to_dict()
                    complete['evaluation_sha256'] = hashlib.sha256(
                        (json.dumps(
                            complete['evaluation'], indent=2, sort_keys=True,
                            allow_nan=False, separators=(',', ': '),
                        ) + '\n').encode('utf-8')
                    ).hexdigest()
                complete_path.chmod(0o644)
                complete_path.write_text(
                    json.dumps(complete, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8',
                )
                args, paths, final_checkpoint, task_classes, _cache = fixture
                with self.assertRaisesRegex(
                        (ValueError, RuntimeError),
                        'seal|cache|complete|inode|hash|identity'):
                    prepare_formal_deferred_evaluation(
                        args=args, snapshot_paths=paths,
                        final_checkpoint=final_checkpoint,
                        task_classes=task_classes, output_dir=tmp,
                    )
                self.assertTrue(
                    (Path(tmp) / 'FORMAL_EVALUATION_SEALED.json').is_file()
                )

    def test_formal_orphan_seal_rejects_before_authorization_or_test_access(self):
        class Dataset:
            def __init__(self, batches):
                self.batches = batches
                self.authorizations = 0
                self.test_accesses = 0

            def authorize_formal_access(self, **_record):
                self.authorizations += 1

            def get_test_loader(self, _classes):
                self.test_accesses += 1
                return iter(self.batches)

        for orphan in ('regular', 'torn', 'symlink', 'wrong_identity'):
            with self.subTest(orphan=orphan), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = self._formal_evaluation_fixture(root)
                seal = root / 'FORMAL_EVALUATION_SEALED.json'
                if orphan == 'regular':
                    seal.write_text('{}\n', encoding='utf-8')
                elif orphan == 'torn':
                    seal.write_text('{', encoding='utf-8')
                elif orphan == 'symlink':
                    target = root / 'orphan-seal-target.json'
                    target.write_text('{}\n', encoding='utf-8')
                    seal.symlink_to(target)
                else:
                    seal.write_text(json.dumps({
                        'schema_version': 1, 'status': 'sealed',
                        'transaction_sha256': '0' * 64,
                        'identity_sha256': '1' * 64,
                        'protocol_sha256': '2' * 64,
                        'freeze': {}, 'cache_identity': {},
                        'pending': {}, 'consuming': {}, 'complete': {},
                    }), encoding='utf-8')
                dataset = Dataset(fixture[-1])
                with mock.patch(
                        'adaptive_consolidation_audit._fresh_trainer',
                        side_effect=lambda _payload, _args:
                        _FormalStateTrainer()):
                    with self.assertRaises(
                            (ValueError, RuntimeError, FileExistsError)):
                        _evaluate_formal_from_single_access(
                            args=fixture[0], dataset=dataset,
                            snapshot_paths=fixture[1],
                            final_checkpoint=fixture[2],
                            task_classes=fixture[3], final_event=1,
                            final_task=1,
                        )
                self.assertEqual(dataset.authorizations, 0)
                self.assertEqual(dataset.test_accesses, 0)
                self.assertFalse(
                    (root / 'FORMAL_EVALUATION_COMPLETE.json').exists()
                )

    def test_formal_post_prevalidation_stage_and_final_replacement_fail_closed(self):
        for target_name in ('stage', 'final'):
            with self.subTest(target=target_name), \
                    tempfile.TemporaryDirectory() as tmp:
                fixture = self._formal_evaluation_fixture(Path(tmp))
                args, paths, final_checkpoint, task_classes, cache = fixture
                prepare_formal_deferred_evaluation(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes, output_dir=tmp,
                )
                target = paths[0] if target_name == 'stage' else final_checkpoint
                original = atomic_write_new_json

                def replace_after_prevalidation(path, payload):
                    installed = original(path, payload)
                    if Path(path).name == 'FORMAL_EVALUATION_CONSUMING.json':
                        replacement = Path(target).with_suffix('.replacement')
                        replacement.write_bytes(Path(target).read_bytes())
                        replacement.replace(target)
                    return installed

                with mock.patch(
                        'adaptive_consolidation_audit.atomic_write_new_json',
                        side_effect=replace_after_prevalidation), mock.patch(
                            'adaptive_consolidation_audit._fresh_trainer',
                            side_effect=lambda _payload, _args:
                            _FormalStateTrainer()):
                    with self.assertRaisesRegex(
                            (ValueError, RuntimeError),
                            'frozen|inode|file|snapshot|checkpoint|identity'):
                        evaluate_formal_deferred_trajectory(
                            args=args, snapshot_paths=paths,
                            final_checkpoint=final_checkpoint,
                            task_classes=task_classes,
                            cached_test_batches=cache, output_dir=tmp,
                        )
                self.assertFalse(
                    (Path(tmp) / 'FORMAL_EVALUATION_COMPLETE.json').exists()
                )

    def test_formal_complete_result_republishes_checkpoint_and_reuses_published_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = list(self._formal_evaluation_fixture(root))
            source = fixture[2]
            formal_final = root / 'checkpoints' / 'formal_final.pt'
            _atomic_torch_save(
                torch.load(source, map_location='cpu', weights_only=True),
                formal_final,
            )
            fixture[2] = formal_final
            fixture = tuple(fixture)
            result, *_ = self._run_formal_evaluation(fixture)
            final = {**result, 'config': {'cl_method': 'er'}}
            with mock.patch(
                    'adaptive_consolidation_audit._fresh_trainer',
                    side_effect=lambda _payload, _args: _FormalStateTrainer()):
                _publish_formal_deferred_result(
                    args=fixture[0], final_checkpoint=formal_final,
                    tracker_state=result, final_result=final,
                )

            published = torch.load(
                formal_final, map_location='cpu', weights_only=True
            )
            provenance = final['source_provenance']
            self.assertEqual(
                published['protocol']['source_provenance'], provenance
            )
            for stage in fixture[1]:
                snapshot = torch.load(
                    stage, map_location='cpu', weights_only=True
                )
                source = torch.load(
                    root / snapshot['source_identity']['path'],
                    map_location='cpu', weights_only=True,
                )
                self.assertEqual(snapshot['source_provenance'], provenance)
                self.assertEqual(
                    snapshot['protocol']['source_provenance'], provenance
                )
                self.assertEqual(
                    source['protocol']['source_provenance'], provenance
                )
            for name in (
                    'FORMAL_STATE_FROZEN.json',
                    'FORMAL_EVALUATION_PENDING.json',
                    'FORMAL_EVALUATION_CONSUMING.json',
                    'FORMAL_EVALUATION_COMPLETE.json'):
                marker = json.loads((root / name).read_text())
                self.assertEqual(
                    marker['identity']['source_provenance'], provenance
                )
            for name in (
                    'FORMAL_EVALUATION_SEALED.json',
                    'FORMAL_EVALUATION_PUBLISHING.json',
                    'FORMAL_EVALUATION_PUBLISHED.json'):
                marker = json.loads((root / name).read_text())
                self.assertEqual(marker['source_provenance'], provenance)

            self.assertEqual(published['schema_version'], 4)
            self.assertEqual(published['tracker_state'], result)
            self.assertEqual(
                json.loads((root / 'results.json').read_text()), final
            )
            self.assertTrue(
                (root / 'FORMAL_EVALUATION_PUBLISHED.json').is_file()
            )
            reused = prepare_formal_deferred_evaluation(
                args=fixture[0], snapshot_paths=fixture[1],
                final_checkpoint=formal_final, task_classes=fixture[3],
                output_dir=tmp,
            )
            self.assertEqual(reused, {'status': 'published', 'result': result})

    def test_formal_results_publication_failure_is_diagnosable_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = list(self._formal_evaluation_fixture(root))
            formal_final = root / 'checkpoints' / 'formal_final.pt'
            _atomic_torch_save(
                torch.load(fixture[2], map_location='cpu', weights_only=True),
                formal_final,
            )
            fixture[2] = formal_final
            fixture = tuple(fixture)
            result, *_ = self._run_formal_evaluation(fixture)
            with mock.patch(
                    'runner._atomic_json_dump',
                    side_effect=RuntimeError('simulated results failure')):
                with self.assertRaisesRegex(RuntimeError, 'results failure'):
                    _publish_formal_deferred_result(
                        args=fixture[0], final_checkpoint=formal_final,
                        tracker_state=result,
                        final_result={**result, 'config': {}},
                    )
            self.assertTrue(
                (root / 'FORMAL_EVALUATION_PUBLISHING.json').is_file()
            )
            self.assertFalse(
                (root / 'FORMAL_EVALUATION_PUBLISHED.json').exists()
            )
            with self.assertRaisesRegex(RuntimeError, 'publication|fail.closed'):
                prepare_formal_deferred_evaluation(
                    args=fixture[0], snapshot_paths=fixture[1],
                    final_checkpoint=formal_final, task_classes=fixture[3],
                    output_dir=tmp,
                )

    def test_formal_publication_rejects_tampered_checkpoint_before_success_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = list(self._formal_evaluation_fixture(root))
            formal_final = root / 'checkpoints' / 'formal_final.pt'
            _atomic_torch_save(
                torch.load(fixture[2], map_location='cpu', weights_only=True),
                formal_final,
            )
            fixture[2] = formal_final
            fixture = tuple(fixture)
            result, *_ = self._run_formal_evaluation(fixture)
            original_save = runner._atomic_torch_save

            def tamper(payload, target):
                forged = copy.deepcopy(payload)
                forged['trainer_state']['top_model']['fallback'] = 99
                return original_save(forged, target)

            with mock.patch(
                    'runner._atomic_torch_save', side_effect=tamper):
                with self.assertRaisesRegex(
                        (ValueError, RuntimeError),
                        'trainer|checkpoint|frozen|identity|reload'):
                    _publish_formal_deferred_result(
                        args=fixture[0], final_checkpoint=formal_final,
                        tracker_state=result,
                        final_result={**result, 'config': {}},
                    )
            self.assertTrue(
                (root / 'FORMAL_EVALUATION_PUBLISHING.json').is_file()
            )
            self.assertFalse(
                (root / 'FORMAL_EVALUATION_PUBLISHED.json').exists()
            )

    def test_formal_publication_reopens_frozen_freeze_and_stage_before_success(self):
        for target_name in ('freeze', 'stage'):
            with self.subTest(target=target_name), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = list(self._formal_evaluation_fixture(root))
                formal_final = root / 'checkpoints' / 'formal_final.pt'
                _atomic_torch_save(
                    torch.load(
                        fixture[2], map_location='cpu', weights_only=True
                    ),
                    formal_final,
                )
                fixture[2] = formal_final
                fixture = tuple(fixture)
                result, *_ = self._run_formal_evaluation(fixture)
                target = (
                    root / 'FORMAL_STATE_FROZEN.json'
                    if target_name == 'freeze' else fixture[1][0]
                )
                original_save = runner._atomic_torch_save

                def replace_after_checkpoint(payload, checkpoint):
                    installed = original_save(payload, checkpoint)
                    replacement = Path(target).with_suffix('.replacement')
                    replacement.write_bytes(Path(target).read_bytes())
                    replacement.replace(target)
                    return installed

                with mock.patch(
                        'runner._atomic_torch_save',
                        side_effect=replace_after_checkpoint), mock.patch(
                            'adaptive_consolidation_audit._fresh_trainer',
                            side_effect=lambda _payload, _args:
                            _FormalStateTrainer()):
                    with self.assertRaisesRegex(
                            (ValueError, RuntimeError),
                            'freeze|snapshot|inode|seal|identity'):
                        _publish_formal_deferred_result(
                            args=fixture[0], final_checkpoint=formal_final,
                            tracker_state=result,
                            final_result={**result, 'config': {}},
                        )
                self.assertTrue(
                    (root / 'FORMAL_EVALUATION_PUBLISHING.json').is_file()
                )
                self.assertFalse(
                    (root / 'FORMAL_EVALUATION_PUBLISHED.json').exists()
                )

    def test_formal_transaction_rejects_torn_symlink_hash_protocol_and_snapshot_tamper(self):
        for mutation in (
                'torn', 'symlink', 'inode', 'hash', 'protocol', 'freeze',
                'snapshot'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = self._formal_evaluation_fixture(Path(tmp))
                args, paths, final_checkpoint, task_classes, _cache = fixture
                prepare_formal_deferred_evaluation(
                    args=args, snapshot_paths=paths,
                    final_checkpoint=final_checkpoint,
                    task_classes=task_classes, output_dir=tmp,
                )
                pending = Path(tmp) / 'FORMAL_EVALUATION_PENDING.json'
                if mutation == 'torn':
                    pending.chmod(0o644)
                    pending.write_text('{', encoding='utf-8')
                elif mutation == 'symlink':
                    saved = pending.with_suffix('.saved')
                    pending.rename(saved)
                    pending.symlink_to(saved)
                elif mutation == 'inode':
                    replacement = final_checkpoint.with_suffix('.replacement')
                    replacement.write_bytes(final_checkpoint.read_bytes())
                    replacement.replace(final_checkpoint)
                elif mutation == 'hash':
                    paths[0].chmod(0o644)
                    paths[0].write_bytes(paths[0].read_bytes() + b'tamper')
                elif mutation == 'protocol':
                    final_checkpoint.chmod(0o644)
                    payload = torch.load(
                        final_checkpoint, map_location='cpu', weights_only=True
                    )
                    payload['protocol']['seed'] += 1
                    torch.save(payload, final_checkpoint)
                elif mutation == 'freeze':
                    freeze = Path(tmp) / 'FORMAL_STATE_FROZEN.json'
                    freeze.chmod(0o644)
                    frozen = json.loads(freeze.read_text())
                    frozen['identity']['source_commit'] = '0' * 40
                    freeze.write_text(json.dumps(frozen), encoding='utf-8')
                else:
                    paths.reverse()
                with self.assertRaises((ValueError, RuntimeError)):
                    prepare_formal_deferred_evaluation(
                        args=args, snapshot_paths=paths,
                        final_checkpoint=final_checkpoint,
                        task_classes=task_classes, output_dir=tmp,
                    )


if __name__ == '__main__':
    unittest.main()
