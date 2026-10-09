import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

import adaptive_head_consolidation as adaptive
import cl_methods.proto_evolve as proto_evolve
import runner
from cl_methods.proto_evolve import ProtoEvolveCL
from cl_methods.proto_evolve_radapt import ProtoEvolveRadaptCL
from cl_methods.sanitize import sanitize_cl_state
from head_consolidation import freeze_state, hash_top_state
from models import TopModel


class TinyTrainer:
    def __init__(self):
        self.bottoms = [nn.Linear(2, 2, bias=False) for _ in range(2)]
        for bottom in self.bottoms:
            with torch.no_grad():
                bottom.weight.copy_(torch.eye(2))
        self.top_model = TopModel(2, 2, cosine=False)
        self.dataset_ref = object()

    @staticmethod
    def _aggregate(embeddings):
        return sum(embeddings)


class RecordingValidation:
    def __init__(self, events, trainer, batch, labels, fail=False):
        self.events = events
        self.trainer = trainer
        self.batch = batch
        self.labels = labels
        self.fail = fail

    def __iter__(self):
        self.events.append('validation_iter')
        with torch.no_grad():
            for bottom in self.trainer.bottoms:
                bottom.weight.add_(10.0)
        if self.fail:
            raise RuntimeError('validation failed')
        yield self.batch, self.labels


class ProtoEvolveAdaptiveHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260814)
        self.batch = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
        ])
        self.labels = torch.tensor([0, 1])

    @staticmethod
    def _args(output_dir, mode='adaptive_dual_branch', num_tasks=2,
              final_ul=False, capacity=20):
        args = SimpleNamespace(
            num_parties=2,
            num_classes=2,
            num_tasks=num_tasks,
            seed=42,
            device='cpu',
            batch_size=2,
            data='synthvfl',
            party_col_ranges=[(0, 2), (2, 4)],
            aggregation='sum',
            output_dir=output_dir,
            head_consolidation_enabled=1,
            head_consolidation_schedule='final',
            head_consolidation_mode=mode,
            head_consolidation_samples_per_class=capacity,
            unlearn_after_tasks=[],
            unlearn_classes=[],
        )
        if final_ul:
            args.unlearn_after_tasks = [num_tasks - 1]
            args.unlearn_classes = [[0]]
        return args

    def _method(self, output_dir, cls=ProtoEvolveCL,
                mode='adaptive_dual_branch', final_ul=False, capacity=20):
        method = cls(
            TinyTrainer(), self._args(output_dir, mode=mode, final_ul=final_ul,
                                      capacity=capacity)
        )
        method.head_raw_replay = {
            0: self.batch[:1].clone(),
            1: self.batch[1:].clone(),
        }
        method.global_protos = {
            0: {'mean': torch.tensor([4.0, 6.0]), 'std': torch.ones(2)},
            1: {'mean': torch.tensor([6.0, 4.0]), 'std': torch.ones(2)},
        }
        method.head_task_classes = {0: [0], 1: [1]}
        return method

    @staticmethod
    def _candidates(pre_top, classes=(0, 1)):
        full = deepcopy(pre_top).eval()
        bias = deepcopy(pre_top).eval()
        with torch.no_grad():
            full.classifier.weight.copy_(torch.tensor([
                [1.5, -0.25], [-0.5, 1.25],
            ]))
            full.classifier.bias.copy_(torch.tensor([0.2, -0.1]))
            bias.classifier.weight.copy_(torch.tensor([
                [0.25, 0.75], [0.8, -0.4],
            ]))
            bias.classifier.bias.copy_(torch.tensor([-0.3, 0.4]))
        bias.set_logit_calibration(
            classes,
            [1.0 + 0.1 * (index + 1) for index in range(len(classes))],
            [0.05 * (index + 1) for index in range(len(classes))],
            list(range(len(classes))),
            1.3,
        )
        return adaptive.FrozenAdaptiveCandidates(
            pre_head_sha256=hash_top_state(pre_top),
            full_state=freeze_state(full.state_dict()),
            bias_state=freeze_state(bias.state_dict()),
            full_head_sha256=hash_top_state(full),
            bias_head_sha256=hash_top_state(bias),
            full_audit={'mode': 'full_classifier'},
            bias_audit={'mode': 'task_class_bias'},
            ordered_classes=tuple(classes),
        )

    @staticmethod
    def _manifest():
        ordered = [5, 9]
        return {
            'dataset': 'fixture-train',
            'seed': 20260814,
            'per_class': 1,
            'by_class': {'0': [5], '1': [9]},
            'ordered_indices': ordered,
            'sha256': hashlib.sha256(json.dumps(
                ordered, separators=(',', ':')
            ).encode('utf-8')).hexdigest(),
        }

    def test_capacity40_uses_version2_for_both_candidate_configs_and_top(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir, capacity=40)
            method.set_head_validation_provider(
                lambda classes: [(self.batch, self.labels)], self._manifest,
            )
            calls = []
            def fit(pre_top, *args, **kwargs):
                calls.append(kwargs)
                return dataclasses.replace(
                    self._candidates(pre_top), samples_per_class=40,
                    method_version=2,
                )
            with mock.patch.object(proto_evolve, 'fit_adaptive_candidates', side_effect=fit):
                result = method._consolidate_head(1)
            self.assertEqual(calls, [{'samples_per_class': 40}])
            self.assertEqual(result.method_version, 2)
            self.assertEqual(result.candidate_configs,
                             adaptive.adaptive_candidate_configs(40))
            self.assertEqual(int(method.trainer.top_model._adaptive_version), 2)
            self.assertEqual(method.get_state()['adaptive_method_version'], 2)
    def test_pre_final_tasks_never_access_validation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir)
            calls = []
            method.set_head_validation_provider(
                lambda classes: calls.append(('loader', tuple(classes))),
                lambda: calls.append(('manifest',)),
            )
            before = hash_top_state(method.trainer.top_model)

            self.assertIsNone(method._consolidate_head(0))

            self.assertEqual(calls, [])
            self.assertEqual(hash_top_state(method.trainer.top_model), before)
            self.assertEqual(method.head_consolidation_history, [])

    def test_final_transaction_freezes_before_validation_and_installs_exact_mixture(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir)
            events = []
            frozen_bottoms = [deepcopy(bottom).eval() for bottom in method.trainer.bottoms]
            parts = [self.batch[:, :2], self.batch[:, 2:]]
            expected_x = sum(bottom(part) for bottom, part in zip(frozen_bottoms, parts))
            validation = RecordingValidation(
                events, method.trainer, self.batch, self.labels,
            )
            method.set_head_validation_provider(
                lambda classes: validation,
                self._manifest,
            )
            solver_inputs = {}

            def fit_spy(pre_top, *args):
                events.extend(['fit_full_frozen', 'fit_bias_frozen'])
                return self._candidates(pre_top)

            def solver_spy(*args, **kwargs):
                events.append('solver')
                self.assertEqual(kwargs, {})
                self.assertEqual(len(args), 4)
                solver_inputs['args'] = args
                return adaptive.solve_global_mixture_weight(*args)

            with mock.patch.object(
                    proto_evolve, 'fit_adaptive_candidates', side_effect=fit_spy
            ), mock.patch.object(
                    proto_evolve, 'solve_global_mixture_weight',
                    side_effect=solver_spy,
            ):
                result = method._consolidate_head(1)

            self.assertLess(events.index('fit_full_frozen'), events.index('validation_iter'))
            self.assertLess(events.index('fit_bias_frozen'), events.index('validation_iter'))
            full_p, bias_p, labels, classes = solver_inputs['args']
            self.assertEqual(classes, (0, 1))
            self.assertTrue(torch.equal(labels, self.labels))
            self.assertEqual(full_p.dtype, torch.float64)
            self.assertEqual(bias_p.dtype, torch.float64)

            installed_full, installed_bias = (
                method.trainer.top_model.branch_log_probabilities(expected_x)
            )
            torch.testing.assert_close(full_p, installed_full, rtol=0, atol=0)
            torch.testing.assert_close(bias_p, installed_bias, rtol=0, atol=0)
            expected = adaptive.mix_log_probabilities(
                installed_full, installed_bias, result.gate['g']
            )
            torch.testing.assert_close(
                method.trainer.top_model(expected_x), expected, rtol=0, atol=0
            )

            Result = getattr(adaptive, 'AdaptiveConsolidationResult', None)
            self.assertIsNotNone(Result)
            self.assertIsInstance(result, Result)
            self.assertEqual(result.candidate_hashes['pre'], result.pre_head_sha256)
            self.assertEqual(result.candidate_configs['full'], adaptive.FULL_BRANCH_CONFIG)
            self.assertEqual(result.candidate_configs['bias'], adaptive.BIAS_BRANCH_CONFIG)
            self.assertEqual(result.validation_manifest['sha256'], self._manifest()['sha256'])
            self.assertEqual(result.task_boundary, 'event_1_CIL')
            self.assertEqual(method.head_validation_sha256, self._manifest()['sha256'])
            self.assertEqual(method.head_consolidation_history, [result.to_dict()])
            json.dumps(dataclasses.asdict(result), allow_nan=False)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                result.task_id = 0
            with self.assertRaises(TypeError):
                result.gate['g'] = 0.0
            self.assertFalse(os.path.exists(os.path.join(output_dir, 'head_consolidation')))

    def test_each_failure_stage_leaves_live_head_and_success_state_unchanged(self):
        stages = ('fit', 'validation', 'solver', 'install', 'commit')
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as output_dir:
                method = self._method(output_dir)
                events = []
                validation = RecordingValidation(
                    events, method.trainer, self.batch, self.labels,
                    fail=stage == 'validation',
                )
                method.set_head_validation_provider(
                    lambda classes: validation,
                    self._manifest,
                )
                before_state = freeze_state(method.trainer.top_model.state_dict())
                before = hash_top_state(before_state)
                candidates = self._candidates(method.trainer.top_model)

                def fit(*args):
                    if stage == 'fit':
                        raise RuntimeError('fit failed')
                    return candidates

                def solve(*args):
                    if stage == 'solver':
                        raise RuntimeError('solver failed')
                    return adaptive.solve_global_mixture_weight(*args)

                def install(*args):
                    if stage == 'install':
                        raise RuntimeError('install failed')
                    return adaptive.install_and_reload_verify(*args)

                live_top = method.trainer.top_model
                top_type = type(live_top)
                live_load = top_type.load_state_dict
                load_calls = 0
                loaded_hashes = []

                def commit_then_fail_once(model, state, *args, **kwargs):
                    nonlocal load_calls
                    result = live_load(model, state, *args, **kwargs)
                    if model is live_top:
                        load_calls += 1
                        loaded_hashes.append(hash_top_state(state))
                        if stage == 'commit' and load_calls == 1:
                            raise RuntimeError('commit failed')
                    return result

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        proto_evolve, 'fit_adaptive_candidates', side_effect=fit,
                    ))
                    stack.enter_context(mock.patch.object(
                        proto_evolve, 'solve_global_mixture_weight', side_effect=solve,
                    ))
                    stack.enter_context(mock.patch.object(
                        proto_evolve, 'install_and_reload_verify', side_effect=install,
                    ))
                    if stage == 'commit':
                        stack.enter_context(mock.patch.object(
                            top_type,
                            'load_state_dict',
                            new=commit_then_fail_once,
                        ))
                    with self.assertRaisesRegex(RuntimeError, f'{stage} failed'):
                        method._consolidate_head(1)

                after_state = method.trainer.top_model.state_dict()
                changed = [
                    key for key in before_state
                    if not torch.equal(before_state[key], after_state[key])
                ]
                self.assertEqual(
                    hash_top_state(method.trainer.top_model), before,
                    (changed, loaded_hashes, before),
                )
                self.assertEqual(method.head_consolidation_history, [])
                self.assertEqual(method.head_validation_sha256, '')
                self.assertFalse(os.path.exists(
                    os.path.join(output_dir, 'head_consolidation')
                ))

    def test_invalid_result_evidence_fails_before_live_commit(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir)
            invalid_manifest = self._manifest()
            invalid_manifest['not_json'] = torch.tensor(1.0)
            method.set_head_validation_provider(
                lambda classes: [(self.batch, self.labels)],
                lambda: invalid_manifest,
            )
            before = hash_top_state(method.trainer.top_model)
            candidates = self._candidates(method.trainer.top_model)

            with mock.patch.object(
                    proto_evolve, 'fit_adaptive_candidates',
                    return_value=candidates,
            ), self.assertRaisesRegex(TypeError, 'strict JSON'):
                method._consolidate_head(1)

            self.assertEqual(hash_top_state(method.trainer.top_model), before)
            self.assertEqual(method.head_consolidation_history, [])
            self.assertEqual(method.head_validation_sha256, '')

    def test_ul_before_final_projects_forgotten_replay_out_before_embedding(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir)
            method.head_raw_replay = {
                0: torch.full((1, 4), -99.0),
                1: self.batch[1:].clone(),
            }
            method.head_task_classes = {0: [0], 1: [1]}
            sanitize_cl_state(method, method.trainer, [0])
            self.assertEqual(tuple(sorted(method.global_protos)), (1,))
            self.assertEqual(tuple(sorted(method.head_raw_replay)), (0, 1))
            seen_bottom_inputs = []
            hooks = [
                bottom.register_forward_pre_hook(
                    lambda module, inputs: seen_bottom_inputs.append(
                        inputs[0].detach().clone()
                    )
                )
                for bottom in method.trainer.bottoms
            ]
            captured = {}

            def fit_spy(pre_top, replay, prototypes, task_classes,
                        raw_count, seed, device):
                captured.update(
                    replay=tuple(sorted(replay)),
                    prototypes=tuple(sorted(prototypes)),
                    task_classes=deepcopy(task_classes),
                    raw_count=raw_count,
                )
                return self._candidates(pre_top, classes=(1,))

            def validation_provider(classes):
                captured['validation_classes'] = tuple(classes)
                return [(self.batch[1:], self.labels[1:])]

            def solve_spy(full, bias, labels, classes):
                captured['solver_classes'] = tuple(classes)
                captured['solver_labels'] = tuple(labels.tolist())
                return adaptive.solve_global_mixture_weight(
                    full, bias, labels, classes
                )

            method.set_head_validation_provider(validation_provider, self._manifest)
            try:
                with mock.patch.object(
                        proto_evolve, 'fit_adaptive_candidates', side_effect=fit_spy
                ), mock.patch.object(
                        proto_evolve, 'solve_global_mixture_weight',
                        side_effect=solve_spy,
                ):
                    result = method._consolidate_head(1)
            finally:
                for hook in hooks:
                    hook.remove()

            self.assertEqual(captured['replay'], (1,))
            self.assertEqual(captured['prototypes'], (1,))
            self.assertEqual(captured['task_classes'], {1: [1]})
            self.assertEqual(captured['raw_count'], 1)
            self.assertEqual(captured['validation_classes'], (1,))
            self.assertEqual(captured['solver_classes'], (1,))
            self.assertEqual(captured['solver_labels'], (1,))
            self.assertEqual(result.ordered_classes, (1,))
            self.assertEqual(
                method.trainer.top_model._adaptive_class_order.tolist(), [1]
            )
            self.assertTrue(seen_bottom_inputs)
            self.assertTrue(all(
                bool((values >= 0).all()) for values in seen_bottom_inputs
            ))

    def test_final_cil_waits_for_final_ul_sanitize_before_validation_and_gate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir, final_ul=True)
            events = []

            def fit_spy(pre_top, replay, prototypes, task_classes,
                        raw_count, seed, device):
                classes = tuple(sorted(replay))
                events.append(('fit', classes, raw_count, deepcopy(task_classes)))
                return self._candidates(pre_top, classes=classes)

            def validation_provider(classes):
                classes = tuple(classes)
                events.append(('validation', classes))
                rows = [index for index, label in enumerate(self.labels.tolist())
                        if label in classes]
                return [(self.batch[rows], self.labels[rows])]

            def solve_spy(full, bias, labels, classes):
                events.append(('solver', tuple(labels.tolist()), tuple(classes)))
                return adaptive.solve_global_mixture_weight(
                    full, bias, labels, classes
                )

            method.set_head_validation_provider(validation_provider, self._manifest)
            with mock.patch.object(
                    proto_evolve, 'fit_adaptive_candidates', side_effect=fit_spy
            ), mock.patch.object(
                    proto_evolve, 'solve_global_mixture_weight',
                    side_effect=solve_spy,
            ):
                self.assertIsNone(method._consolidate_head(1))
                self.assertEqual(events, [])
                self.assertFalse(bool(method.trainer.top_model._adaptive_enabled))
                self.assertEqual(
                    method.get_state()['adaptive_pending_task_id'], 1
                )
                method.global_protos = deepcopy(method.global_protos)
                self.assertEqual(events, [])
                self.assertEqual(method._adaptive_pending_task_id, 1)

                runner._sanitize_and_finalize_adaptive(
                    method, method.trainer, [0], adaptive_mode=True
                )

            self.assertEqual(events, [
                ('fit', (1,), 1, {1: [1]}),
                ('validation', (1,)),
                ('solver', (1,), (1,)),
            ])
            self.assertEqual(
                method.trainer.top_model._adaptive_class_order.tolist(), [1]
            )
            inference = method.trainer.top_model(torch.tensor([[4.0, 6.0]]))
            self.assertEqual(inference.shape, torch.Size([1, 1]))
            self.assertTrue(torch.isfinite(inference).all())
            self.assertEqual(
                method.head_consolidation_history[-1]['ordered_classes'], [1]
            )
            self.assertIsNone(
                method.get_state()['adaptive_pending_task_id']
            )

    def test_runner_never_finalizes_when_sanitizer_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir, final_ul=True)
            validation_accesses = []
            method.set_head_validation_provider(
                lambda classes: validation_accesses.append(tuple(classes)),
                self._manifest,
            )
            method._consolidate_head(1)
            before = hash_top_state(method.trainer.top_model)

            with mock.patch(
                    'cl_methods.sanitize.sanitize_cl_state',
                    side_effect=RuntimeError('sanitize failed'),
            ), self.assertRaisesRegex(RuntimeError, 'sanitize failed'):
                runner._sanitize_and_finalize_adaptive(
                    method, method.trainer, [0], adaptive_mode=True
                )

            self.assertEqual(validation_accesses, [])
            self.assertEqual(hash_top_state(method.trainer.top_model), before)
            self.assertEqual(method.head_consolidation_history, [])
            self.assertEqual(method._adaptive_pending_task_id, 1)

    def test_forged_earlier_pending_callback_fails_without_triggering(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = self._args(output_dir, num_tasks=2)
            args.unlearn_after_tasks = [0, 1]
            args.unlearn_classes = [[0], [1]]
            method = ProtoEvolveCL(TinyTrainer(), args)
            method._adaptive_pending_task_id = 0
            validation_accesses = []
            method.set_head_validation_provider(
                lambda classes: validation_accesses.append(tuple(classes)),
                self._manifest,
            )
            before = hash_top_state(method.trainer.top_model)

            with self.assertRaisesRegex(ValueError, 'pending task'):
                method.finalize_adaptive_head_after_sanitize()

            self.assertEqual(validation_accesses, [])
            self.assertEqual(hash_top_state(method.trainer.top_model), before)
            self.assertEqual(method.head_consolidation_history, [])
            self.assertEqual(method._adaptive_pending_task_id, 0)

    def test_adaptive_mode_requires_state_sanitization_but_fixed_does_not(self):
        with tempfile.TemporaryDirectory() as output_dir:
            adaptive_args = self._args(output_dir)
            adaptive_args.sanitize_cl_state = 0
            with self.assertRaisesRegex(ValueError, 'sanitize_cl_state'):
                ProtoEvolveCL(TinyTrainer(), adaptive_args)

            fixed_args = self._args(output_dir, mode='full_classifier')
            fixed_args.sanitize_cl_state = 0
            ProtoEvolveCL(TinyTrainer(), fixed_args)

    def test_pending_checkpoint_double_load_never_runs_transaction(self):
        with tempfile.TemporaryDirectory() as output_dir:
            source = self._method(output_dir, final_ul=True)
            source._consolidate_head(1)
            state = source.get_state()

            restored = self._method(output_dir, final_ul=True)
            validation_accesses = []

            def forbidden_validation(classes):
                validation_accesses.append(tuple(classes))
                raise AssertionError('pending resume accessed validation')

            restored.set_head_validation_provider(
                forbidden_validation, self._manifest
            )
            before = hash_top_state(restored.trainer.top_model)
            candidates = self._candidates(restored.trainer.top_model)
            with mock.patch.object(
                    proto_evolve, 'fit_adaptive_candidates',
                    return_value=candidates,
            ):
                restored.load_state(deepcopy(state))
                restored.load_state(deepcopy(state))

            self.assertEqual(validation_accesses, [])
            self.assertEqual(hash_top_state(restored.trainer.top_model), before)
            self.assertEqual(restored.head_consolidation_history, [])
            self.assertEqual(restored._adaptive_pending_task_id, 1)

    def test_metadata_publication_failure_rolls_back_live_top_and_history(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = self._method(output_dir)
            publisher = getattr(method, '_publish_adaptive_metadata', None)
            self.assertIsNotNone(
                publisher, 'transactional metadata publisher is missing'
            )
            method.set_head_validation_provider(
                lambda classes: [(self.batch, self.labels)], self._manifest,
            )
            before = hash_top_state(method.trainer.top_model)
            candidates = self._candidates(method.trainer.top_model)

            def fail_after_partial_publish(history, validation_hash):
                method.head_consolidation_history = history
                raise RuntimeError('metadata publication failed')

            with mock.patch.object(
                    proto_evolve, 'fit_adaptive_candidates',
                    return_value=candidates,
            ), mock.patch.object(
                    method, '_publish_adaptive_metadata',
                    side_effect=fail_after_partial_publish,
            ), self.assertRaisesRegex(RuntimeError, 'metadata publication failed'):
                method._consolidate_head(1)

            self.assertEqual(hash_top_state(method.trainer.top_model), before)
            self.assertEqual(method.head_consolidation_history, [])
            self.assertEqual(method.head_validation_sha256, '')

    def test_adaptive_provider_excludes_inherited_dataset_paths_but_fixed_does_not(self):
        with tempfile.TemporaryDirectory() as output_dir:
            adaptive_method = self._method(output_dir, cls=ProtoEvolveRadaptCL)
            adaptive_method.set_head_validation_provider(lambda classes: (), self._manifest)
            self.assertIsNone(adaptive_method.trainer.dataset_ref)
            with mock.patch.object(
                    ProtoEvolveCL, 'after_task', return_value=None
            ), mock.patch.object(adaptive_method, '_refresh_redundancy') as refresh:
                adaptive_method.after_task([], 0)
            refresh.assert_not_called()

            fixed_method = self._method(output_dir, mode='full_classifier')
            inherited = fixed_method.trainer.dataset_ref
            fixed_method.set_head_validation_provider(lambda classes: (), self._manifest)
            self.assertIs(fixed_method.trainer.dataset_ref, inherited)


if __name__ == '__main__':
    unittest.main()
