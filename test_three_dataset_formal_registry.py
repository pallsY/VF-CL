import os
import tempfile
import unittest
from unittest import mock

from config import get_config
import three_dataset_formal_registry as registry

from three_dataset_formal_registry import (
    FormalSpec,
    _normalize_options,
    command_for,
    explanation_specs,
    formal_specs,
    parsed_protocol,
    protocol_for,
    registry_sha256,
    safe_spec_name,
    validation_access_for,
)


class ThreeDatasetFormalRegistryTest(unittest.TestCase):
    def test_continuation_has_exact_old_scientific_protocols(self):
        old_env = {'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                   'VFCL_FORMAL_DATASET': 'cifar100'}
        new_env = {'VFCL_EXPERIMENT_PROFILE':
                   'single-dataset-verified-continuation-v1',
                   'VFCL_FORMAL_DATASET': 'cifar100'}
        with mock.patch.dict(os.environ, old_env):
            old = {registry.registered_spec_key(spec):
                   registry.protocol_for(spec)
                   for spec in registry.formal_specs()}
        with mock.patch.dict(os.environ, new_env):
            self.assertEqual(registry.profile_cardinality(), (42, 0))
            self.assertEqual({spec.dataset for spec in registry.formal_specs()},
                             {'cifar100'})
            current = {registry.registered_spec_key(spec):
                       registry.protocol_for(spec)
                       for spec in registry.formal_specs()}
        self.assertEqual(current, old)
        self.assertEqual(len(current), 42)
        for dataset in ('isolet', 'upmc_food101'):
            with mock.patch.dict(os.environ, {**new_env,
                                              'VFCL_FORMAL_DATASET': dataset}):
                with self.assertRaises(ValueError):
                    registry.experiment_profile()

    def test_method_contract_binds_every_formal_runtime_identity(self):
        self.assertTrue(hasattr(registry, 'method_contract_for'))
        expected = {
            'finetune': ('finetune', False, 'full_classifier'),
            'lwf': ('lwf', False, 'full_classifier'),
            'gpm': ('gpm', False, 'full_classifier'),
            'fedprotip_vfl': ('fedprotip_vfl', False, 'full_classifier'),
            'er': ('er', False, 'full_classifier'),
            'no_consolidation': ('proto_evolve', False, 'full_classifier'),
            'fixed_full': ('proto_evolve', True, 'adaptive_dual_branch'),
            'fixed_bias': ('proto_evolve', True, 'adaptive_dual_branch'),
            'adaptive': ('proto_evolve', True, 'adaptive_dual_branch'),
            'fixed_half': ('proto_evolve', True, 'adaptive_dual_branch'),
            'sample_mean_nll': (
                'proto_evolve', True, 'adaptive_dual_branch'),
        }
        for spec in (*formal_specs(), *explanation_specs()):
            with self.subTest(spec=spec):
                contract = registry.method_contract_for(spec)
                self.assertEqual(
                    expected[spec.method],
                    (contract['cl_method'],
                     contract['head_consolidation_enabled'],
                     contract['head_consolidation_mode']),
                )
                self.assertEqual(
                    contract, protocol_for(spec)['method_contract'])
                self.assertTrue(all(
                    protocol_for(spec)['base_options'][key] == value
                    for key, value in contract.items()
                ))

        er = registry.method_contract_for(FormalSpec('isolet', 'er', 42))
        self.assertEqual((20, 64, 1.0), (
            er['er_per_class'], er['er_batch'], er['er_alpha']))
        lwf = registry.method_contract_for(FormalSpec('isolet', 'lwf', 42))
        self.assertEqual((2.0, 0.5, 1.0, True), (
            lwf['lwf_temperature'], lwf['lwf_alpha'], lwf['lwf_lambda'],
            lwf['lwf_ce_newonly']))
        gpm = registry.method_contract_for(FormalSpec('isolet', 'gpm', 42))
        self.assertEqual(0.95, gpm['gpm_threshold'])
        tip = registry.method_contract_for(
            FormalSpec('isolet', 'fedprotip_vfl', 42))
        self.assertEqual((0.775, 20), (
            tip['fedprotip_tip_threshold'], tip['fedprotip_max_batches']))
        fixed = registry.method_contract_for(
            FormalSpec('cifar100', 'fixed_full', 42))
        self.assertEqual((0.01, 500), (
            fixed['head_consolidation_lr'],
            fixed['head_consolidation_steps'],
        ))
        bias = registry.method_contract_for(
            FormalSpec('isolet', 'fixed_bias', 42))
        self.assertEqual((0.03, 600), (
            bias['head_consolidation_lr'],
            bias['head_consolidation_steps'],
        ))
        self.assertEqual(
            'class_balanced',
            registry.method_contract_for(
                FormalSpec('upmc_food101', 'adaptive', 42)
            )['head_gate_rule'],
        )
        self.assertEqual(
            ('fixed_half_ablation', 'sample_mean_ablation'),
            tuple(registry.method_contract_for(
                FormalSpec('isolet', method, 42, True)
            )['head_gate_rule'] for method in (
                'fixed_half', 'sample_mean_nll')),
        )
        self.assertTrue(all(
            not set(registry.method_contract_for(spec)).intersection(
                registry._SHARED_VALIDATION_OPTIONS)
            for spec in (*formal_specs(), *explanation_specs())
        ))

    def test_exact_matrix(self):
        self.assertEqual('formal', registry.experiment_profile())
        self.assertEqual(
            'seed42-adaptive-recovery', registry.RECOVERY_PROFILE)
        formal = formal_specs()
        explanation = explanation_specs()
        self.assertEqual((81, 81), (len(formal), len(set(formal))))
        self.assertEqual((6, 6), (len(explanation), len(set(explanation))))
        self.assertEqual({42, 43, 44}, {spec.seed for spec in formal})

    def test_single_dataset_matrix_has_only_42_frozen_cells(self):
        for dataset in registry.DATASETS:
            with self.subTest(dataset=dataset), mock.patch.dict(os.environ, {
                    'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                    'VFCL_FORMAL_DATASET': dataset}):
                specs = formal_specs()
                self.assertEqual(42, len(specs))
                self.assertEqual(42, len(set(specs)))
                self.assertEqual({dataset}, {spec.dataset for spec in specs})
                self.assertEqual(set(registry.FULL_MATRIX_METHODS),
                                 {spec.method for spec in specs})
                self.assertEqual({42, 43, 44}, {spec.seed for spec in specs})
                self.assertEqual((), explanation_specs())
                self.assertEqual((42, 0), registry.profile_cardinality())

    def test_single_dataset_selector_is_required_and_profile_scoped(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix'},
                clear=True), self.assertRaises(ValueError):
            formal_specs()
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
                'VFCL_FORMAL_DATASET': 'unknown'}), self.assertRaises(ValueError):
            formal_specs()
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix',
                'VFCL_FORMAL_DATASET': 'isolet'}), self.assertRaises(ValueError):
            formal_specs()

    def test_full_public_matrix_is_exactly_126_cells(self):
        expected_methods = (
            'finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
            'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
            'proto_fedspace', 'adaptive',
        )
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: 'full-public-matrix'}):
            specs = registry.formal_specs()
            self.assertEqual((126, 0), registry.profile_cardinality())
            self.assertEqual(126, len(specs))
            self.assertEqual(126, len(set(specs)))
            self.assertEqual(
                expected_methods,
                tuple(dict.fromkeys(spec.method for spec in specs)),
            )
            self.assertEqual({42, 43, 44}, {spec.seed for spec in specs})
            self.assertEqual(
                set(registry.DATASETS), {spec.dataset for spec in specs})
            self.assertEqual((), registry.explanation_specs())
            self.assertFalse({
                'lwf_fim', 'proto_evolve_radapt', 'proto_aug', 'prl',
            } & {spec.method for spec in specs})

    def test_full_public_matrix_method_contracts_are_exact(self):
        common = {
            'dep_tracking_enabled': False,
            'party_kd_enabled': False,
            'head_consolidation_enabled': False,
            'head_consolidation_mode': 'full_classifier',
        }
        method_options = {
            'finetune': {},
            'lwf': dict(lwf_temperature=2.0, lwf_alpha=0.5,
                        lwf_lambda=1.0, lwf_ce_newonly=True),
            'ewc': dict(ewc_lambda=1000.0, ewc_fisher_decay=0.9,
                        ewc_fisher_samples=1024, lwf_ce_newonly=True,
                        feat_distill_weight=0.0),
            'er': dict(er_per_class=20, er_batch=64, er_alpha=1.0),
            'der_pp': dict(der_buffer_size=0, der_batch=64,
                           der_alpha=0.5, der_beta=0.5),
            'er_ace': dict(er_ace_buffer_size=0, er_ace_batch=64),
            'gpm': dict(gpm_threshold=0.95),
            'fedprotip_vfl': dict(fedprotip_tip_threshold=0.775,
                                  fedprotip_max_batches=20),
            'target': {},
            'afc': dict(afc_distill_weight=2.0),
            'lwf_wa': dict(lwf_temperature=2.0, lwf_alpha=0.5,
                           lwf_lambda=1.0, lwf_ce_newonly=True),
            'adagauss': dict(adagauss_lambda_ac=0.2,
                             adagauss_lambda_pkd=1.0,
                             adagauss_shrinkage=0.1,
                             adagauss_adapter_epochs=30,
                             adagauss_n_samples=256),
            'proto_fedspace': dict(proto_aug_weight=1.0,
                                   repr_loss_weight=0.1),
        }
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.FULL_MATRIX_PROFILE}):
            self.assertEqual(
                set(method_options),
                set(registry.FULL_MATRIX_METHODS).intersection(
                    registry.EXTERNAL_METHODS),
            )
            for method, options in method_options.items():
                with self.subTest(method=method):
                    self.assertEqual(
                        {'cl_method': method, **common, **options},
                        registry.method_contract_for(
                            FormalSpec('isolet', method, 42)),
                    )
            self.assertEqual(
                'proto_evolve',
                registry.method_contract_for(
                    FormalSpec('isolet', 'adaptive', 42))['cl_method'],
            )

    def test_full_public_matrix_er_ace_budget_is_exact_and_method_specific(self):
        expected = {'er_ace_buffer_size': 0, 'er_ace_batch': 64}
        self.assertEqual(
            {key: 'int' for key in expected},
            {key: registry.OPTION_SCHEMA.get(key) for key in expected},
        )
        with mock.patch.dict(os.environ, {
                registry.PROFILE_ENV: registry.FULL_MATRIX_PROFILE}):
            specs = formal_specs()
            er_ace_specs = tuple(
                spec for spec in specs if spec.method == 'er_ace')
            self.assertEqual(
                (set(registry.DATASETS), set(registry.FULL_MATRIX_SEEDS), 9),
                ({spec.dataset for spec in er_ace_specs},
                 {spec.seed for spec in er_ace_specs}, len(er_ace_specs)),
            )
            for spec in specs:
                with self.subTest(spec=spec):
                    contract = registry.method_contract_for(spec)
                    actual = {
                        key: contract[key] for key in expected
                        if key in contract
                    }
                    self.assertEqual(
                        expected if spec.method == 'er_ace' else {}, actual)
                    command = command_for(spec, 'cuda:0', '/result')
                    start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                    raw = registry._option_map(command, start)
                    raw_budget = {
                        key: int(raw[key]) for key in expected if key in raw
                    }
                    self.assertEqual(
                        expected if spec.method == 'er_ace' else {},
                        raw_budget,
                    )
                    self.assertEqual(
                        protocol_for(spec), parsed_protocol(command, spec))

    def test_er_ace_formal_parser_binds_budget_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--cl_method', 'er_ace',
                    '--formal_deferred_evaluation', '1',
                    '--results_dir', tmp, '--exp_name', 'valid-er-ace',
                    '--er_ace_buffer_size', '0', '--er_ace_batch', '64']):
                args = get_config()
            self.assertEqual((0, 64), (
                args.er_ace_buffer_size, args.er_ace_batch))

        invalid = (
            ('--er_ace_buffer_size', '-1',
             'formal ER-ACE requires --er_ace_buffer_size >= 0'),
            ('--er_ace_batch', '0',
             'formal ER-ACE requires --er_ace_batch > 0'),
        )
        for option, value, message in invalid:
            with self.subTest(option=option), \
                    tempfile.TemporaryDirectory() as tmp:
                exp_name = f'invalid-{option[2:]}'
                with mock.patch('sys.argv', [
                        'main.py', '--cl_method', 'er_ace',
                        '--formal_deferred_evaluation', '1',
                        '--results_dir', tmp, '--exp_name', exp_name,
                        option, value]), mock.patch('sys.stderr') as stderr, \
                        self.assertRaises(SystemExit) as raised:
                    get_config()
                self.assertEqual(2, raised.exception.code)
                self.assertIn(message, ''.join(
                    call.args[0] for call in stderr.write.call_args_list))
                self.assertFalse(os.path.lexists(os.path.join(tmp, exp_name)))

    def test_er_ace_flags_are_rejected_for_every_other_formal_method(self):
        methods = (
            'finetune', 'proto_aug', 'proto_evolve', 'proto_fedspace',
            'der_pp', 'er', 'ewc', 'lwf', 'target', 'gpm',
            'fedprotip_vfl', 'prl', 'afc', 'lwf_fim', 'lwf_wa',
            'adagauss', 'proto_evolve_radapt',
        )
        explicit_cases = (
            (('--er_ace_buffer_size=0',), '--er_ace_buffer_size'),
            (('--er_ace_batch', '64'), '--er_ace_batch'),
            (('--er_ace_bat', '64'), '--er_ace_batch'),
            (('--er_ace_bat=17',), '--er_ace_batch'),
            (('--er_ace_buffer', '-1'), '--er_ace_buffer_size'),
        )
        for method in methods:
            for tokens, option in explicit_cases:
                with self.subTest(method=method, option=option), \
                        tempfile.TemporaryDirectory() as tmp:
                    exp_name = f'invalid-er-ace-option-{method}'
                    with mock.patch('sys.argv', [
                            'main.py', '--cl_method', method,
                            '--formal_deferred_evaluation', '1',
                            '--results_dir', tmp, '--exp_name', exp_name,
                            *tokens]), mock.patch('sys.stderr') as stderr, \
                            self.assertRaises(SystemExit) as raised:
                        get_config()
                    self.assertEqual(2, raised.exception.code)
                    self.assertIn(
                        f'ER-ACE option requires --cl_method er_ace: {option}',
                        ''.join(call.args[0]
                                for call in stderr.write.call_args_list),
                    )
                    self.assertFalse(
                        os.path.lexists(os.path.join(tmp, exp_name)))

        for method in methods:
            with self.subTest(method=method, case='defaults'), \
                    tempfile.TemporaryDirectory() as tmp:
                exp_name = f'valid-with-er-ace-defaults-{method}'
                with mock.patch('sys.argv', [
                        'main.py', '--cl_method', method,
                        '--formal_deferred_evaluation', '1',
                        '--results_dir', tmp, '--exp_name', exp_name]):
                    args = get_config()
                self.assertEqual(
                    (0, 64), (args.er_ace_buffer_size, args.er_ace_batch))
                self.assertTrue(os.path.isfile(
                    os.path.join(tmp, exp_name, 'config.json')))

    def test_er_ace_abbreviations_parse_then_use_exact_range_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('sys.argv', [
                    'main.py', '--cl_method', 'er_ace',
                    '--formal_deferred_evaluation', '1',
                    '--results_dir', tmp, '--exp_name', 'valid-abbreviated',
                    '--er_ace_buffer', '0', '--er_ace_bat=17']):
                args = get_config()
            self.assertEqual(
                (0, 17), (args.er_ace_buffer_size, args.er_ace_batch))

        invalid = (
            (('--er_ace_buffer', '-1'),
             'formal ER-ACE requires --er_ace_buffer_size >= 0'),
            (('--er_ace_bat=0',),
             'formal ER-ACE requires --er_ace_batch > 0'),
        )
        for tokens, message in invalid:
            with self.subTest(tokens=tokens), \
                    tempfile.TemporaryDirectory() as tmp:
                with mock.patch('sys.argv', [
                        'main.py', '--cl_method', 'er_ace',
                        '--formal_deferred_evaluation', '1',
                        '--results_dir', tmp,
                        '--exp_name', 'invalid-abbreviated',
                        *tokens]), mock.patch('sys.stderr') as stderr, \
                        self.assertRaises(SystemExit) as raised:
                    get_config()
                self.assertEqual(2, raised.exception.code)
                self.assertIn(message, ''.join(
                    call.args[0] for call in stderr.write.call_args_list))
                self.assertFalse(os.path.lexists(
                    os.path.join(tmp, 'invalid-abbreviated')))

    def test_er_ace_options_are_always_stripped_from_inherited_base_flags(self):
        inherited = {
            'seed': '42',
            'er_ace_buffer_size': '999',
            'er_ace_batch': '999',
        }
        for profile in (
                registry.FORMAL_PROFILE, registry.PILOT_PROFILE,
                registry.RECOVERY_PROFILE):
            with self.subTest(profile=profile), mock.patch.dict(os.environ, {
                    registry.PROFILE_ENV: profile}):
                self.assertEqual(
                    {'seed': '42'},
                    registry._without_inherited_method_options(inherited),
                )

    def test_safe_spec_names_are_injective_and_drive_exact_output_directories(self):
        specs = (*formal_specs(), *explanation_specs())
        names = [safe_spec_name(spec) for spec in specs]
        self.assertEqual(len(specs), len(set(names)))
        self.assertTrue(all(
            name and name not in {'.', '..'}
            and '/' not in name and '\\' not in name
            and '..' not in name
            for name in names
        ))
        for spec, name in zip(specs, names):
            with self.subTest(spec=spec):
                command = command_for(spec, 'cuda:0', '/formal/runs')
                index = command.index('--exp_name')
                self.assertEqual(name, command[index + 1])
                self.assertEqual(
                    f'/formal/runs/{name}',
                    os.path.join(command[command.index('--results_dir') + 1],
                                 command[index + 1]),
                )
                self.assertEqual(protocol_for(spec), parsed_protocol(command, spec))

    def test_safe_spec_name_and_parser_reject_traversal_or_collision_inputs(self):
        for spec in (
                FormalSpec('../isolet', 'finetune', 42),
                FormalSpec('isolet', '../finetune', 42),
                FormalSpec('isolet', 'finetune', -1),
                FormalSpec('isolet', 'fixed_half', 43, True)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                safe_spec_name(spec)
        spec = FormalSpec('isolet', 'finetune', 42)
        command = list(command_for(spec, 'cuda:0', '/formal/runs'))
        command[command.index('--exp_name') + 1] = '../collision'
        with self.assertRaises(ValueError):
            parsed_protocol(tuple(command), spec)

    def test_validation_seeds(self):
        expected = {
            'cifar100': 20260729,
            'isolet': 20260809,
            'upmc_food101': 20260809,
        }
        for spec in formal_specs():
            self.assertEqual(
                expected[spec.dataset], protocol_for(spec)['validation_split_seed']
            )
        self.assertEqual(
            '977fa501de5980ac43c1ed6113dca0a4391658488805794a243a2dcc80db807b',
            registry_sha256(),
        )

    def test_all_commands_share_the_dataset_holdout_contract(self):
        holdout_names = (
            'lambda_validation_enabled',
            'lambda_validation_per_class',
            'lambda_validation_split_seed',
        )
        for spec in (*formal_specs(), *explanation_specs()):
            with self.subTest(spec=spec):
                expected = {
                    'lambda_validation_enabled': True,
                    'lambda_validation_per_class':
                        registry.VALIDATION_PER_CLASS[spec.dataset],
                    'lambda_validation_split_seed':
                        registry.VALIDATION_SPLIT_SEEDS[spec.dataset],
                }
                protocol = protocol_for(spec)
                self.assertEqual(
                    expected,
                    {name: protocol['base_options'][name]
                     for name in holdout_names},
                )
                self.assertFalse(
                    set(protocol['method_contract']).intersection(
                        holdout_names))
                command = command_for(spec, 'cuda:0', '/result')
                start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                raw = registry._option_map(command, start)
                self.assertEqual(
                    {'lambda_validation_enabled': '1',
                     'lambda_validation_per_class':
                         str(registry.VALIDATION_PER_CLASS[spec.dataset]),
                     'lambda_validation_split_seed':
                         str(registry.VALIDATION_SPLIT_SEEDS[spec.dataset])},
                    {name: raw[name] for name in holdout_names},
                )
                self.assertIs(
                    validation_access_for(spec),
                    spec.method not in registry.EXTERNAL_METHODS
                    and spec.method != 'no_consolidation',
                )

    def test_all_fedprotip_commands_emit_and_require_exact_raw_options(self):
        specs = tuple(
            spec for spec in formal_specs()
            if spec.method == 'fedprotip_vfl'
        )
        self.assertEqual(9, len(specs))
        expected = {
            '--fedprotip_tip_threshold': '0.775',
            '--fedprotip_max_batches': '20',
        }
        for spec in specs:
            with self.subTest(spec=spec):
                command = command_for(spec, 'cuda:0', '/result')
                raw = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(
                    expected,
                    {option: raw[option] for option in expected},
                )
                for option, value in expected.items():
                    with self.subTest(option=option, case='missing'):
                        missing = list(command)
                        if option in missing:
                            index = missing.index(option, 2)
                            del missing[index:index + 2]
                        with self.assertRaises(ValueError):
                            parsed_protocol(tuple(missing), spec)
                    with self.subTest(option=option, case='conflicting'):
                        conflicting = list(command)
                        if option in conflicting:
                            conflicting[conflicting.index(option, 2) + 1] = (
                                '0.5' if option.endswith('threshold') else '21')
                        else:
                            conflicting.extend((option, '0.5'))
                        with self.assertRaises(ValueError):
                            parsed_protocol(tuple(conflicting), spec)
                    with self.subTest(option=option, case='duplicated'):
                        with self.assertRaises(ValueError):
                            parsed_protocol((*command, option, value), spec)

    def test_protocol_freezes_normalized_base_options(self):
        cifar = protocol_for(FormalSpec('cifar100', 'finetune', 42))['base_options']
        self.assertEqual(4, cifar['num_parties'])
        self.assertEqual('sgd', cifar['optimizer'])
        self.assertIs(True, cifar['deterministic'])
        vector = protocol_for(FormalSpec('isolet', 'finetune', 42))['base_options']
        self.assertEqual((0, 1), vector['custom_tasks'][0])
        with self.assertRaisesRegex(ValueError, 'unexpected frozen option'):
            _normalize_options({'unregistered_option': '1'})

    def test_all_commands_parse_to_the_exact_frozen_protocol(self):
        specs = (*formal_specs(), *explanation_specs())
        for spec in specs:
            with self.subTest(spec=spec):
                command = command_for(spec, 'cuda:7', '/deployment/result')
                self.assertIsInstance(command, tuple)
                self.assertEqual(protocol_for(spec), parsed_protocol(command, spec))
                joined = ' '.join(command).lower()
                self.assertNotIn('tinyimagenet', joined)
                self.assertNotIn('official-val', joined)
        self.assertEqual(87, len(specs))
        self.assertEqual(87, len({
            command_for(spec, 'cuda:0', '/result') for spec in specs
        }))

    def test_all_commands_bind_formal_protocol_through_real_config_parser(self):
        specs = (*formal_specs(), *explanation_specs())
        identities = []
        with tempfile.TemporaryDirectory() as temporary_results:
            for spec in specs:
                with self.subTest(spec=spec):
                    results_dir = os.path.join(
                        temporary_results,
                        f'{spec.dataset}-{spec.method}-{spec.seed}-'
                        f'{int(spec.explanation)}',
                    )
                    command = command_for(spec, 'cuda:7', results_dir)
                    start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                    self.assertEqual(
                        1, command[start:].count('--formal_deferred_evaluation'))
                    with mock.patch.dict(
                            os.environ,
                            {'VFCL_REVIEWED_ADAPTIVE_ABLATION': '1'}), \
                            mock.patch(
                                'sys.argv', ['main.py', *command[start:]]):
                        args = get_config()
                    self.assertIs(args.formal_deferred_evaluation, True)
                    self.assertEqual(args.fedprotip_tip_threshold, 0.775)
                    self.assertEqual(args.fedprotip_max_batches, 20)
                    identities.append((
                        spec.dataset, spec.method, spec.seed, spec.explanation,
                        args.cl_method, args.seed,
                    ))
        self.assertEqual(87, len(identities))
        self.assertEqual(87, len(set(identities)))

    def test_all_commands_require_one_exact_formal_deferred_flag(self):
        for spec in (*formal_specs(), *explanation_specs()):
            with self.subTest(spec=spec):
                command = list(command_for(spec, 'cuda:0', '/result'))
                start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                option = '--formal_deferred_evaluation'
                index = command.index(option, start)
                missing = command[:index] + command[index + 2:]
                conflicting = list(command)
                conflicting[index + 1] = '0'
                for case in (missing, conflicting, command + [option, '1']):
                    with self.assertRaises(ValueError):
                        parsed_protocol(tuple(case), spec)

    def test_commands_use_one_exact_option_set_and_reviewed_entrypoint(self):
        root = str(registry.Path(registry.__file__).resolve().parent)
        external = {'finetune', 'lwf', 'gpm', 'fedprotip_vfl', 'er'}
        for spec in (*formal_specs(), *explanation_specs()):
            with self.subTest(spec=spec):
                command = command_for(spec, 'cuda:0', '/result')
                start = 2 if spec.method in external else 3
                options = command[start:]
                self.assertEqual(0, len(options) % 2)
                names = options[0::2]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(name.startswith('--') for name in names))
                if spec.method in external:
                    self.assertEqual(
                        str(registry.Path(root) / 'main.py'), command[1])
                    self.assertNotEqual('-c', command[1])
                else:
                    self.assertEqual('-c', command[1])
                    self.assertEqual(
                        'from three_dataset_formal_runtime import '
                        'execute_variant_main; '
                        f'execute_variant_main({spec.method!r})',
                        command[2],
                    )

    def test_commands_use_the_reviewed_3080_deployment_paths(self):
        expected_python = (
            '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python')
        with mock.patch.dict(
                os.environ, {'VFCL_PYTHON': expected_python}, clear=True):
            root, worktree, python = registry._deployment_paths()
            self.assertEqual(expected_python, str(python))
            self.assertTrue(python.is_file())
            self.assertTrue(worktree.is_dir())
            self.assertTrue((worktree / 'main.py').is_file())
            for spec in (*formal_specs(), *explanation_specs()):
                with self.subTest(spec=spec):
                    command = command_for(spec, 'cuda:0', '/result')
                    self.assertEqual(expected_python, command[0])
                    self.assertNotIn('/home/chase', ' '.join(command))
                    start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                    options = registry._option_map(command, start)
                    data_path = registry.Path(options['data_path'])
                    self.assertTrue(data_path.is_dir())
                    self.assertEqual(root, data_path.parent)
                    if 'vector_npz' in options:
                        vector = registry.Path(options['vector_npz'])
                        self.assertTrue(vector.is_file())
                        self.assertEqual(root, vector.parents[2])
                    self.assertEqual(protocol_for(spec), parsed_protocol(command, spec))

        for environment, error in (({}, RuntimeError),
                                   ({'VFCL_PYTHON': '/usr/bin/python'}, ValueError)):
            with self.subTest(environment=environment), \
                    mock.patch.dict(os.environ, environment, clear=True), \
                    self.assertRaises(error):
                command_for(FormalSpec('isolet', 'finetune', 42), 'cuda:0', '/result')

    def test_parser_requires_raw_nonempty_device_and_results_pairs(self):
        for spec in (FormalSpec('isolet', 'er', 42),
                     FormalSpec('cifar100', 'fixed_full', 42)):
            with self.subTest(spec=spec):
                command = command_for(spec, 'cuda:0', '/result')
                start = 2 if spec.method in registry.EXTERNAL_METHODS else 3
                for option in ('--device', '--results_dir'):
                    with self.subTest(option=option, case='missing'):
                        missing = list(command)
                        index = missing.index(option, start)
                        del missing[index:index + 2]
                        with self.assertRaises(ValueError):
                            parsed_protocol(tuple(missing), spec)
                    with self.subTest(option=option, case='empty'):
                        empty = list(command)
                        empty[empty.index(option, start) + 1] = ''
                        with self.assertRaises(ValueError):
                            parsed_protocol(tuple(empty), spec)

    def test_smoke_is_a_nonreturning_exact_bool_boundary(self):
        spec = FormalSpec('isolet', 'finetune', 42)
        for smoke in (0, 1, 'False', None):
            with self.subTest(smoke=smoke), self.assertRaises(TypeError):
                command_for(spec, 'cuda:0', '/result', smoke=smoke)
        with self.assertRaises(ValueError):
            command_for(spec, 'cuda:0', '/result', smoke=True)
        for current in (*formal_specs(), *explanation_specs()):
            with self.subTest(spec=current):
                command = command_for(current, 'cuda:0', '/result', smoke=False)
                self.assertEqual(protocol_for(current), parsed_protocol(command, current))

    def test_formal_spec_fields_require_exact_primitives(self):
        invalid = (
            FormalSpec(1, 'adaptive', 42),
            FormalSpec('isolet', 1, 42),
            FormalSpec('isolet', 'adaptive', True),
            FormalSpec('isolet', 'adaptive', 42, 1),
        )
        for spec in invalid:
            with self.subTest(spec=spec):
                with self.assertRaises(TypeError):
                    registry.method_contract_for(spec)
                with self.assertRaises(TypeError):
                    protocol_for(spec)

    def test_command_construction_is_repeatable_and_call_order_independent(self):
        specs = (*formal_specs(), *explanation_specs())
        forward = {
            spec: command_for(spec, 'cuda:0', '/result') for spec in specs
        }
        reverse = {
            spec: command_for(spec, 'cuda:0', '/result')
            for spec in reversed(specs)
        }
        self.assertEqual(forward, reverse)
        self.assertTrue(all(
            command == command_for(spec, 'cuda:0', '/result')
            for spec, command in forward.items()
        ))

    def test_parser_rejects_extra_duplicate_conflicting_and_wrong_entrypoints(self):
        external = FormalSpec('isolet', 'er', 42)
        command = command_for(external, 'cuda:0', '/result')
        with self.assertRaises(ValueError):
            parsed_protocol((*command, '--unknown-option', '1'), external)
        with self.assertRaises(ValueError):
            parsed_protocol((*command, '--seed', '43'), external)
        conflict = list(command)
        conflict[conflict.index('--er_per_class') + 1] = '21'
        with self.assertRaises(ValueError):
            parsed_protocol(tuple(conflict), external)
        wrong_main = list(command)
        wrong_main[1] = '/tmp/main.py'
        with self.assertRaises(ValueError):
            parsed_protocol(tuple(wrong_main), external)

        internal = FormalSpec('cifar100', 'fixed_full', 42)
        wrapped = list(command_for(internal, 'cuda:0', '/result'))
        wrapped[2] = wrapped[2].replace('fixed_full', 'fixed_bias')
        with self.assertRaises(ValueError):
            parsed_protocol(tuple(wrapped), internal)

    def test_command_rejects_unregistered_specs_and_invalid_deployment_fields(self):
        invalid = FormalSpec('cifar100', 'fixed_half', 43, True)
        with self.assertRaises(ValueError):
            command_for(invalid, 'cuda:0', '/result')
        spec = FormalSpec('cifar100', 'adaptive', 42)
        for device, destination in (('', '/result'), ('cuda:0', '')):
            with self.subTest(device=device, destination=destination), \
                    self.assertRaises(ValueError):
                command_for(spec, device, destination)

    def test_er_command_freezes_exact_twenty_per_class_contract(self):
        for dataset in registry.DATASETS:
            command = command_for(
                FormalSpec(dataset, 'er', 42), 'cuda:0', '/result')
            start = command.index('--cl_method')
            flags = dict(zip(command[start::2], command[start + 1::2]))
            self.assertEqual('er', flags['--cl_method'])
            self.assertEqual('20', flags['--er_per_class'])
            self.assertEqual('64', flags['--er_batch'])
            self.assertEqual('1.0', flags['--er_alpha'])
