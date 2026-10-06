import copy
import contextlib
import gc
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import mock

import data_utils
import three_dataset_formal_registry as registry
from cl_methods import get_cl_method
from cl_methods.afc import AFCCL
from cl_methods.er import ERCL
from cl_methods.finetune import FineTuneCL
from cl_methods.lwf import LwFCL
from cl_methods.proto_evolve import ProtoEvolveCL
from config import get_config
from models import build_models
from three_dataset_formal_runtime import variant_runtime
from vfl_trainer import VFLTrainer


DATASET_FIELDS = (
    'data',
    'vector_npz',
    'num_classes',
    'num_tasks',
    'custom_tasks',
    'classes_per_task',
    'num_parties',
    'model_type',
    'aggregation',
    'epochs_per_task',
    'batch_size',
    'num_workers',
    'optimizer',
    'lr',
    'bottom_lr_scale',
    'momentum',
    'weight_decay',
    'task_ce_mode',
    'deterministic',
    'data_flow_audit',
    'save_task_checkpoints',
    'formal_deferred_evaluation',
)

ISOLET_TASKS = tuple((label, label + 1) for label in range(0, 26, 2))
UPMC_TASKS = (
    tuple(range(0, 11)),
    tuple(range(11, 21)),
    tuple(range(21, 31)),
    tuple(range(31, 41)),
    tuple(range(41, 51)),
    tuple(range(51, 61)),
    tuple(range(61, 71)),
    tuple(range(71, 81)),
    tuple(range(81, 91)),
    tuple(range(91, 101)),
)
EXPECTED_DATASET_CONTRACTS = {
    'cifar100': {
        'data': 'cifar100',
        'vector_npz': None,
        'num_classes': 100,
        'num_tasks': 10,
        'custom_tasks': None,
        'classes_per_task': 10,
        'num_parties': 4,
        'model_type': 'resnet18',
        'aggregation': 'sum',
        'epochs_per_task': 50,
        'batch_size': 64,
        'num_workers': 2,
        'optimizer': 'sgd',
        'lr': 0.001,
        'bottom_lr_scale': 1.0,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'task_ce_mode': 'method',
        'deterministic': True,
        'data_flow_audit': True,
        'save_task_checkpoints': 3,
        'formal_deferred_evaluation': True,
    },
    'isolet': {
        'data': 'tabvfl',
        'vector_npz': 'isolet/isolet_vfl.npz',
        'num_classes': 26,
        'num_tasks': 13,
        'custom_tasks': ISOLET_TASKS,
        'classes_per_task': 2,
        'num_parties': 4,
        'model_type': 'mlp',
        'aggregation': 'concat',
        'epochs_per_task': 50,
        'batch_size': 128,
        'num_workers': 2,
        'optimizer': 'adamw',
        'lr': 0.001,
        'bottom_lr_scale': 1.0,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'task_ce_mode': 'current',
        'deterministic': True,
        'data_flow_audit': True,
        'save_task_checkpoints': 3,
        'formal_deferred_evaluation': True,
    },
    'upmc_food101': {
        'data': 'tabvfl',
        'vector_npz': 'upmc_food101/upmc_food101_vfl.npz',
        'num_classes': 101,
        'num_tasks': 10,
        'custom_tasks': UPMC_TASKS,
        'classes_per_task': 11,
        'num_parties': 2,
        'model_type': 'mlp',
        'aggregation': 'concat',
        'epochs_per_task': 20,
        'batch_size': 128,
        'num_workers': 2,
        'optimizer': 'adamw',
        'lr': 0.003,
        'bottom_lr_scale': 0.25,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'task_ce_mode': 'current',
        'deterministic': True,
        'data_flow_audit': True,
        'save_task_checkpoints': 3,
        'formal_deferred_evaluation': True,
    },
}

ABLATION_DELTAS = {
    'no_consolidation': {
        'head_consolidation_enabled',
        'head_consolidation_mode',
    },
    'fixed_full': set(),
    'fixed_bias': {
        'head_consolidation_lr',
        'head_consolidation_steps',
    },
    'fixed_half': {'head_gate_rule'},
    'sample_mean_nll': {'head_gate_rule'},
}

EXPECTED_PILOT_METHOD_CLASSES = {
    'finetune': FineTuneCL,
    'lwf': LwFCL,
    'er': ERCL,
    'afc': AFCCL,
    'adaptive': ProtoEvolveCL,
}
REVIEWED_PYTHON = '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'


def _spec(dataset, method, seed):
    return registry.FormalSpec(
        dataset, method, seed, method in registry.EXPLANATIONS
    )


def _assert_named_method_contract_is_identical(test_case, method, seed):
    contracts = [
        registry.method_contract_for(_spec(dataset, method, seed))
        for dataset in registry.DATASETS
    ]
    for contract in contracts[1:]:
        test_case.assertEqual(contracts[0], contract)


def _dataset_projection(spec):
    options = registry.protocol_for(spec)['base_options']
    return {field: options.get(field) for field in DATASET_FIELDS}


def _assert_dataset_contract_is_consistent(test_case, dataset, seed):
    methods = tuple(
        spec.method for spec in registry.formal_specs()
        if spec.dataset == dataset and spec.seed == seed)
    projections = [
        _dataset_projection(_spec(dataset, method, seed))
        for method in methods
    ]
    for projection in projections[1:]:
        test_case.assertEqual(projections[0], projection)
    portable = dict(projections[0])
    vector_npz = portable['vector_npz']
    if vector_npz is not None:
        portable['vector_npz'] = '/'.join(Path(vector_npz).parts[-2:])
    test_case.assertEqual(EXPECTED_DATASET_CONTRACTS[dataset], portable)


def _assert_constructed_method_class_is_identical(test_case, classes):
    reference = classes[registry.DATASETS[0]]
    for dataset in registry.DATASETS[1:]:
        test_case.assertIs(reference, classes[dataset])


class ThreeDatasetMethodContractTests(unittest.TestCase):
    def test_named_method_contract_is_identical_across_datasets(self):
        self.assertEqual(3, len(registry.DATASETS))
        self.assertEqual(9, len(registry.METHODS))
        self.assertEqual((42, 43, 44), registry.SEEDS)
        self.assertEqual(2, len(registry.EXPLANATIONS))
        for seed in registry.SEEDS:
            for method in registry.METHODS:
                with self.subTest(method=method, seed=seed):
                    _assert_named_method_contract_is_identical(
                        self, method, seed
                    )
        for method in registry.EXPLANATIONS:
            with self.subTest(method=method, seed=42):
                _assert_named_method_contract_is_identical(self, method, 42)

    def test_each_dataset_uses_one_exact_model_and_training_contract(self):
        for dataset, seed in sorted({
                (spec.dataset, spec.seed) for spec in registry.formal_specs()}):
            with self.subTest(dataset=dataset, seed=seed):
                _assert_dataset_contract_is_consistent(self, dataset, seed)

    def test_ablation_family_has_only_the_approved_contract_deltas(self):
        adaptive = registry.method_contract_for(
            registry.FormalSpec('cifar100', 'adaptive', 42)
        )
        variants = ('adaptive', *ABLATION_DELTAS)
        for variant, expected_delta in ABLATION_DELTAS.items():
            with self.subTest(variant=variant):
                contract = registry.method_contract_for(
                    _spec('cifar100', variant, 42)
                )
                keys = set(adaptive) | set(contract)
                changed = {
                    key for key in keys
                    if adaptive.get(key) != contract.get(key)
                }
                self.assertEqual(expected_delta, changed)
                for key in keys - expected_delta:
                    self.assertEqual(adaptive.get(key), contract.get(key))

        args = SimpleNamespace(
            num_parties=1,
            num_classes=2,
            num_tasks=1,
            device='cpu',
            head_consolidation_mode='adaptive_dual_branch',
            head_consolidation_schedule='final',
            sanitize_cl_state=1,
        )
        for variant in variants:
            with self.subTest(constructed_variant=variant):
                contract = registry.method_contract_for(
                    _spec('cifar100', variant, 42)
                )
                instance = get_cl_method(contract['cl_method'], None, args)
                self.assertIs(type(instance), ProtoEvolveCL)

    def test_seed42_pilot_real_cpu_construction_for_all_cells(self):
        if registry.experiment_profile() != registry.PILOT_PROFILE:
            self.skipTest('real construction matrix belongs to seed42-pilot')
        self.assertEqual(
            set(EXPECTED_PILOT_METHOD_CLASSES), set(registry.PILOT_METHODS))
        specs = registry.formal_specs()
        self.assertEqual(15, len(specs))
        original_cifar100 = data_utils.datasets.CIFAR100
        cifar_download_calls = []

        def existing_only_cifar100(*args, **kwargs):
            requested = kwargs.get('download', False)
            forwarded = {**kwargs, 'download': False}
            cifar_download_calls.append((requested, forwarded['download']))
            return original_cifar100(*args, **forwarded)

        expected_bottoms = {
            'cifar100': (4, 'ResNet18Bottom'),
            'isolet': (4, 'MLPBottom'),
            'upmc_food101': (2, 'MLPBottom'),
        }
        with TemporaryDirectory(prefix='vfcl-method-consistency-') as directory:
            temporary_root = Path(directory).resolve()
            outputs = []
            for spec in specs:
                dataset_name = spec.dataset
                with mock.patch.dict(
                        os.environ, {'VFCL_PYTHON': REVIEWED_PYTHON}):
                    command = registry.command_for(
                        spec, 'cpu', str(temporary_root))
                option_start = command.index('--data')
                argv = ['main.py', *command[option_start:]]
                runtime = (variant_runtime('adaptive')
                           if spec.method == 'adaptive'
                           else contextlib.nullcontext())
                with runtime, mock.patch.object(
                        sys, 'argv', argv):
                    args = get_config()
                    if dataset_name == 'cifar100':
                        with mock.patch.object(
                                data_utils.datasets, 'CIFAR100',
                                side_effect=existing_only_cifar100):
                            dataset = data_utils.VFLDataset(args)
                        self.assertTrue(any(
                            requested
                            for requested, _forwarded in cifar_download_calls
                        ))
                        self.assertTrue(all(
                            not forwarded
                            for _requested, forwarded in cifar_download_calls
                        ))
                    else:
                        dataset = data_utils.VFLDataset(args)
                    bottoms, top = build_models(args)
                    trainer = VFLTrainer(bottoms, top, args)
                    method = get_cl_method(args.cl_method, trainer, args)

                expected_count, expected_class = expected_bottoms[dataset_name]
                self.assertEqual('cpu', args.device)
                self.assertEqual(expected_count, len(bottoms))
                self.assertEqual(
                    [expected_class] * expected_count,
                    [type(bottom).__name__ for bottom in bottoms],
                )
                self.assertEqual(args.num_classes, top.classifier.out_features)
                self.assertEqual(args.num_parties, len(bottoms))
                self.assertIs(
                    type(method), EXPECTED_PILOT_METHOD_CLASSES[spec.method])

                output = Path(args.output_dir).resolve()
                self.assertEqual(
                    temporary_root,
                    Path(os.path.commonpath((temporary_root, output))),
                )
                outputs.append(output)
                del method, trainer, top, bottoms, dataset, args
                gc.collect()

            self.assertEqual(len(specs), len(set(outputs)))
            self.assertTrue(all(output.is_dir() for output in outputs))

        self.assertFalse(temporary_root.exists())
        self.assertFalse(any(output.exists() for output in outputs))

    def test_cross_dataset_helper_detects_seed43_isolet_parameter_drift(self):
        original = registry.method_contract_for

        def drifted_contract(spec):
            contract = original(spec)
            if (spec.dataset == 'isolet' and spec.method == 'adaptive'
                    and spec.seed == 43):
                contract = {**contract, 'party_kd_lambda': 9.0}
            return contract

        with mock.patch.object(
                registry, 'method_contract_for', side_effect=drifted_contract):
            with self.assertRaises(AssertionError):
                _assert_named_method_contract_is_identical(self, 'adaptive', 43)

    def test_dataset_helper_detects_one_registered_cifar_method_model_drift(self):
        original = registry.protocol_for

        def drifted_protocol(spec):
            protocol = copy.deepcopy(original(spec))
            if spec.dataset == 'cifar100' and spec.method == 'finetune':
                protocol['base_options']['model_type'] = 'mlp'
            return protocol

        with mock.patch.object(
                registry, 'protocol_for', side_effect=drifted_protocol):
            with self.assertRaises(AssertionError):
                _assert_dataset_contract_is_consistent(self, 'cifar100', 42)

    def test_method_class_helper_detects_upmc_different_class(self):
        class DifferentMethod:
            pass

        classes = {dataset: ProtoEvolveCL for dataset in registry.DATASETS}
        classes['upmc_food101'] = DifferentMethod
        with self.assertRaises(AssertionError):
            _assert_constructed_method_class_is_identical(self, classes)


if __name__ == '__main__':
    unittest.main()
