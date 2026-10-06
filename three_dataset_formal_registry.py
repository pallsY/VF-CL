from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import fair_main_table_3datasets as fair
import unified_head_consolidation_factorial as factorial


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
FORMAL_PROFILE = 'formal'
PILOT_PROFILE = 'seed42-pilot'
RECOVERY_PROFILE = 'seed42-adaptive-recovery'
FULL_MATRIX_PROFILE = 'full-public-matrix'
SINGLE_DATASET_PROFILE = 'single-dataset-full-matrix'
CONTINUATION_PROFILE = 'single-dataset-verified-continuation-v1'
METHOD_SHARD_PROFILE = 'single-method-formal-v1'
_DATASET_SCOPED_PROFILES = (
    SINGLE_DATASET_PROFILE, CONTINUATION_PROFILE, METHOD_SHARD_PROFILE)
PROFILE_ENV = 'VFCL_EXPERIMENT_PROFILE'
DATASET_ENV = 'VFCL_FORMAL_DATASET'
METHOD_ENV = 'VFCL_FORMAL_METHOD'
METHODS = (
    'finetune',
    'lwf',
    'gpm',
    'fedprotip_vfl',
    'er',
    'no_consolidation',
    'fixed_full',
    'fixed_bias',
    'adaptive',
)
SEEDS = (42, 43, 44)
EXPLANATIONS = ('fixed_half', 'sample_mean_nll')
PILOT_METHODS = ('finetune', 'lwf', 'er', 'afc', 'adaptive')
PILOT_SEEDS = (42,)
RECOVERY_METHODS = ('adaptive',)
RECOVERY_SEEDS = (42,)
FULL_MATRIX_METHODS = (
    'finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
    'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
    'proto_fedspace', 'adaptive',
)
FULL_MATRIX_SEEDS = (42, 43, 44)
EXTERNAL_METHODS = frozenset(
    ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace', 'gpm',
     'fedprotip_vfl', 'target', 'afc', 'lwf_wa', 'adagauss',
     'proto_fedspace'))
VALIDATION_SPLIT_SEEDS = {
    'cifar100': 20260729,
    'isolet': 20260809,
    'upmc_food101': 20260809,
}
VALIDATION_PER_CLASS = {
    'cifar100': 25,
    'isolet': 40,
    'upmc_food101': 64,
}
OPTION_SCHEMA = {
    'data': 'str',
    'data_path': 'path',
    'vector_npz': 'path',
    'num_classes': 'int',
    'num_tasks': 'int',
    'custom_tasks': 'tasks',
    'classes_per_task': 'int',
    'unlearn_after_tasks': 'int',
    'unlearn_classes': 'int',
    'num_parties': 'int',
    'model_type': 'str',
    'aggregation': 'str',
    'epochs_per_task': 'int',
    'batch_size': 'int',
    'num_workers': 'int',
    'optimizer': 'str',
    'lr': 'float',
    'bottom_lr_scale': 'float',
    'task_ce_mode': 'str',
    'momentum': 'float',
    'weight_decay': 'float',
    'device': 'str',
    'ul_method': 'str',
    'replay_mode': 'str',
    'deterministic': 'bool',
    'formal_deferred_evaluation': 'bool',
    'data_flow_audit': 'bool',
    'bic_enabled': 'bool',
    'bic_fit_mode': 'str',
    'bic_lr': 'float',
    'bic_per_class': 'int',
    'bic_split_seed': 'int',
    'bic_steps': 'int',
    'lambda_validation_enabled': 'bool',
    'lambda_validation_per_class': 'int',
    'lambda_validation_split_seed': 'int',
    'der_buffer_size': 'int',
    'der_batch': 'int',
    'er_ace_buffer_size': 'int',
    'er_ace_batch': 'int',
    'ewc_lambda': 'float',
    'ewc_fisher_decay': 'float',
    'ewc_fisher_samples': 'int',
    'der_alpha': 'float',
    'der_beta': 'float',
    'adagauss_lambda_ac': 'float',
    'adagauss_lambda_pkd': 'float',
    'adagauss_shrinkage': 'float',
    'adagauss_adapter_epochs': 'int',
    'adagauss_n_samples': 'int',
    'proto_aug_weight': 'float',
    'repr_loss_weight': 'float',
    'save_task_checkpoints': 'int',
    'seed': 'int',
    'cl_method': 'str',
    'dep_tracking_enabled': 'bool',
    'party_kd_enabled': 'bool',
    'party_kd_mode': 'str',
    'expected_party_kd_variant': 'str',
    'party_kd_lambda': 'float',
    'proto_replay_loss_norm': 'str',
    'proto_replay_ratio': 'float',
    'proto_lambda_a': 'float',
    'fim_freeze_frac': 'float',
    'head_consolidation_enabled': 'bool',
    'head_consolidation_mode': 'str',
    'head_consolidation_regularization': 'float',
    'head_consolidation_lr': 'float',
    'head_consolidation_steps': 'int',
    'head_consolidation_class_regularization': 'float',
    'head_consolidation_task_regularization': 'float',
    'head_consolidation_task_weight': 'float',
    'head_consolidation_samples_per_class': 'int',
    'head_consolidation_schedule': 'str',
    'head_full_lr': 'float',
    'head_full_steps': 'int',
    'head_bias_lr': 'float',
    'head_bias_steps': 'int',
    'head_gate_rule': 'str',
    'head_gate_solver_tolerance': 'float',
    'head_gate_solver_max_iterations': 'int',
    'distill_weight': 'float',
    'feat_distill_weight': 'float',
    'current_supcon_weight': 'float',
    'lwf_temperature': 'float',
    'lwf_alpha': 'float',
    'lwf_lambda': 'float',
    'lwf_ce_newonly': 'text_bool',
    'gpm_threshold': 'float',
    'fedprotip_tip_threshold': 'float',
    'fedprotip_max_batches': 'int',
    'afc_distill_weight': 'float',
    'er_per_class': 'int',
    'er_batch': 'int',
    'er_alpha': 'float',
    'results_dir': 'path',
    'exp_name': 'str',
}
_SHARED_VALIDATION_OPTIONS = frozenset((
    'lambda_validation_enabled',
    'lambda_validation_per_class',
    'lambda_validation_split_seed',
))
_VALIDATION_ACCESS_METHODS = frozenset({
    'fixed_full', 'fixed_bias', 'adaptive',
    'fixed_half', 'sample_mean_nll',
})


@dataclass(frozen=True, order=True)
class FormalSpec:
    dataset: str
    method: str
    seed: int
    explanation: bool = False


def experiment_profile():
    value = os.environ.get(PROFILE_ENV, FORMAL_PROFILE)
    if value not in {
            FORMAL_PROFILE, PILOT_PROFILE, RECOVERY_PROFILE,
            FULL_MATRIX_PROFILE, *_DATASET_SCOPED_PROFILES}:
        raise RuntimeError(f'unsupported experiment profile: {value}')
    dataset = os.environ.get(DATASET_ENV)
    method = os.environ.get(METHOD_ENV)
    if value == METHOD_SHARD_PROFILE:
        if dataset != 'cifar100':
            raise ValueError('method shard profile requires CIFAR-100')
        if method not in FULL_MATRIX_METHODS:
            raise ValueError('method shard requires a registered method')
    elif value == CONTINUATION_PROFILE:
        if dataset != 'cifar100':
            raise ValueError('continuation profile requires CIFAR-100')
    elif value == SINGLE_DATASET_PROFILE:
        if dataset not in DATASETS:
            raise ValueError('single-dataset profile requires a reviewed dataset')
    elif dataset is not None:
        raise ValueError('dataset selector requires a dataset-scoped profile')
    if value != METHOD_SHARD_PROFILE and method is not None:
        raise ValueError('formal method selector requires method-shard profile')
    return value


def selected_formal_dataset():
    if experiment_profile() not in _DATASET_SCOPED_PROFILES:
        raise ValueError('dataset-scoped profile is required')
    return os.environ[DATASET_ENV]


def selected_formal_method():
    if experiment_profile() != METHOD_SHARD_PROFILE:
        raise ValueError('method-shard profile is required')
    return os.environ[METHOD_ENV]


def profile_cardinality():
    profile = experiment_profile()
    if profile == PILOT_PROFILE:
        return 15, 0
    if profile == RECOVERY_PROFILE:
        return 3, 0
    if profile == FULL_MATRIX_PROFILE:
        return 126, 0
    if profile in _DATASET_SCOPED_PROFILES:
        if profile == METHOD_SHARD_PROFILE:
            return 3, 0
        return 42, 0
    return 81, 6


def _profile_members():
    profile = experiment_profile()
    if profile == PILOT_PROFILE:
        return PILOT_METHODS, PILOT_SEEDS, ()
    if profile == RECOVERY_PROFILE:
        return RECOVERY_METHODS, RECOVERY_SEEDS, ()
    if profile == METHOD_SHARD_PROFILE:
        return (selected_formal_method(),), FULL_MATRIX_SEEDS, ()
    if profile == FULL_MATRIX_PROFILE or profile in _DATASET_SCOPED_PROFILES:
        return FULL_MATRIX_METHODS, FULL_MATRIX_SEEDS, ()
    return METHODS, SEEDS, EXPLANATIONS


experiment_profile()


_REVIEWED_VFCL_PYTHON = Path(
    '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python')


def _validate_spec(spec):
    if not isinstance(spec, FormalSpec):
        raise TypeError('spec must be a FormalSpec')
    for name, expected_type in (
            ('dataset', str), ('method', str), ('seed', int),
            ('explanation', bool)):
        if type(getattr(spec, name)) is not expected_type:
            raise TypeError(f'spec.{name} must be exactly {expected_type.__name__}')


def _holdout_contract(spec):
    return {
        'lambda_validation_enabled': True,
        'lambda_validation_per_class': VALIDATION_PER_CLASS[spec.dataset],
        'lambda_validation_split_seed': VALIDATION_SPLIT_SEEDS[spec.dataset],
    }


def _deployment_paths():
    root, worktree, _ = factorial.deployment_paths(module_file=__file__)
    python = os.environ.get('VFCL_PYTHON')
    if python is None:
        raise RuntimeError('VFCL_PYTHON must name the reviewed Python interpreter')
    if python != str(_REVIEWED_VFCL_PYTHON):
        raise ValueError('VFCL_PYTHON differs from the reviewed Python interpreter')
    python = Path(python)
    if not python.is_file():
        raise RuntimeError(f'VFCL_PYTHON is not a file: {python}')
    return root, worktree, python


def _deployment_option_values(spec, root):
    values = {'data_path': str(root / 'data')}
    if spec.dataset != 'cifar100':
        values['vector_npz'] = str(
            root / 'data' / spec.dataset / f'{spec.dataset}_vfl.npz')
    return values


def _option_map(command, start=2):
    tokens = tuple(str(token) for token in command[start:])
    if len(tokens) % 2:
        raise ValueError('frozen command contains an unpaired option')
    pairs = tuple(zip(tokens[0::2], tokens[1::2]))
    if any(not option.startswith('--') for option, _ in pairs):
        raise ValueError('frozen command contains a positional argument')
    if len({option for option, _ in pairs}) != len(pairs):
        raise ValueError('frozen command repeats an option')
    return {option[2:]: value for option, value in pairs}


def _normalize_value(kind, value):
    if kind == 'bool':
        if value not in ('0', '1'):
            raise ValueError(f'invalid frozen boolean: {value}')
        return value == '1'
    if kind == 'text_bool':
        if value not in ('False', 'True'):
            raise ValueError(f'invalid frozen text boolean: {value}')
        return value == 'True'
    if kind == 'int':
        return int(value)
    if kind == 'float':
        return float(value)
    if kind == 'tasks':
        return tuple(
            tuple(int(label) for label in task.split(','))
            for task in value.split('|')
        )
    if kind == 'path':
        return str(Path(value))
    if kind == 'str':
        return str(value)
    raise ValueError(f'unknown frozen option schema: {kind}')


def _normalize_options(options):
    unexpected = sorted(set(options).difference(OPTION_SCHEMA))
    if unexpected:
        raise ValueError(f'unexpected frozen option: {unexpected[0]}')
    return {
        option: _normalize_value(OPTION_SCHEMA[option], value)
        for option, value in options.items()
    }


def _base_command(spec, device, results_dir, smoke):
    if spec.dataset == 'cifar100':
        return factorial.cifar_base_command(device, results_dir, smoke)
    return fair.base_command(
        f'{spec.dataset}:finetune:{spec.seed}', device, results_dir,
        smoke=smoke,
    )


def _method_option_names():
    methods, seeds, explanations = _profile_members()
    if experiment_profile() == METHOD_SHARD_PROFILE:
        methods, seeds, explanations = (
            FULL_MATRIX_METHODS, FULL_MATRIX_SEEDS, ())
    representatives = (
        *(FormalSpec(DATASETS[0], method, seeds[0]) for method in methods),
        *(FormalSpec(DATASETS[0], method, seeds[0], True)
          for method in explanations),
    )
    return (
        {key for spec in representatives for key in method_contract_for(spec)
         if key in OPTION_SCHEMA}
        .difference(_SHARED_VALIDATION_OPTIONS)
        .union((
            'der_buffer_size', 'der_batch',
            'er_ace_buffer_size', 'er_ace_batch',
        ))
    )


def _without_inherited_method_options(options):
    options = dict(options)
    for option in _method_option_names():
        options.pop(option, None)
    return options


def _base_options(spec):
    options = _without_inherited_method_options(
        _option_map(_base_command(spec, 'cuda:0', '/result', False)))
    options['seed'] = str(spec.seed)
    options['formal_deferred_evaluation'] = '1'
    options['exp_name'] = safe_spec_name(spec)
    normalized = _normalize_options(options)
    normalized.update(method_contract_for(spec))
    normalized.update(_holdout_contract(spec))
    return normalized


def method_contract_for(spec: FormalSpec) -> dict:
    """Return the canonical effective runtime identity for one formal method."""
    _validate_spec(spec)
    if spec.method in EXTERNAL_METHODS and not spec.explanation:
        contract = {
            'cl_method': spec.method,
            'dep_tracking_enabled': False,
            'party_kd_enabled': False,
            'head_consolidation_enabled': False,
            'head_consolidation_mode': 'full_classifier',
        }
        if spec.method in {'lwf', 'lwf_wa'}:
            contract.update(
                lwf_temperature=2.0, lwf_alpha=0.5, lwf_lambda=1.0,
                lwf_ce_newonly=True,
            )
        elif spec.method == 'ewc':
            contract.update(
                ewc_lambda=1000.0,
                ewc_fisher_decay=0.9,
                ewc_fisher_samples=1024,
                lwf_ce_newonly=True,
                feat_distill_weight=0.0,
            )
        elif spec.method == 'der_pp':
            contract.update(
                der_buffer_size=0,
                der_batch=64,
                der_alpha=0.5,
                der_beta=0.5,
            )
        elif spec.method == 'er_ace':
            contract.update(er_ace_buffer_size=0, er_ace_batch=64)
        elif spec.method == 'gpm':
            contract['gpm_threshold'] = 0.95
        elif spec.method == 'fedprotip_vfl':
            contract.update(
                fedprotip_tip_threshold=0.775,
                fedprotip_max_batches=20,
            )
        elif spec.method == 'er':
            contract.update(
                er_per_class=(300 if experiment_profile() == PILOT_PROFILE
                              else 20),
                er_batch=64,
                er_alpha=1.0,
            )
        elif spec.method == 'afc':
            contract['afc_distill_weight'] = 2.0
        elif spec.method == 'adagauss':
            contract.update(
                adagauss_lambda_ac=0.2,
                adagauss_lambda_pkd=1.0,
                adagauss_shrinkage=0.1,
                adagauss_adapter_epochs=30,
                adagauss_n_samples=256,
            )
        elif spec.method == 'proto_fedspace':
            contract.update(proto_aug_weight=1.0, repr_loss_weight=0.1)
        return contract

    if spec.method not in {
            'no_consolidation', 'fixed_full', 'fixed_bias', 'adaptive',
            'fixed_half', 'sample_mean_nll'}:
        raise ValueError(f'unregistered method contract: {spec.method}')
    mode = ('full_classifier' if spec.method == 'no_consolidation'
            else 'adaptive_dual_branch')
    lr, steps = ((0.03, 600) if spec.method == 'fixed_bias'
                 else (0.01, 500))
    gate = {
        'fixed_half': 'fixed_half_ablation',
        'sample_mean_nll': 'sample_mean_ablation',
    }.get(spec.method, 'class_balanced')
    return {
        'cl_method': 'proto_evolve',
        'dep_tracking_enabled': True,
        'party_kd_enabled': True,
        'party_kd_mode': 'uniform',
        'expected_party_kd_variant': 'uniform',
        'party_kd_lambda': 1.0,
        'proto_replay_loss_norm': 'sample_mean',
        'proto_replay_ratio': 1.0,
        'proto_lambda_a': 0.15,
        'fim_freeze_frac': 0.0,
        'head_consolidation_enabled': spec.method != 'no_consolidation',
        'head_consolidation_mode': mode,
        'head_consolidation_regularization': 0.01,
        'head_consolidation_lr': lr,
        'head_consolidation_steps': steps,
        'head_consolidation_class_regularization': 0.01,
        'head_consolidation_task_regularization': 0.01,
        'head_consolidation_task_weight': 1.3,
        'head_consolidation_samples_per_class': 20,
        'head_consolidation_schedule': 'final',
        'head_full_lr': 0.01,
        'head_full_steps': 500,
        'head_bias_lr': 0.03,
        'head_bias_steps': 600,
        'head_gate_rule': gate,
        'head_gate_solver_tolerance': 1e-12,
        'head_gate_solver_max_iterations': 80,
        'distill_weight': 0.25,
        'feat_distill_weight': 0.05,
        'current_supcon_weight': 0.0,
    }


def formal_specs() -> tuple[FormalSpec, ...]:
    methods, seeds, _ = _profile_members()
    if experiment_profile() == PILOT_PROFILE:
        return tuple(
            FormalSpec(dataset, method, seed)
            for seed in seeds
            for method in methods
            for dataset in DATASETS
        )
    datasets = ((selected_formal_dataset(),)
                if experiment_profile() in _DATASET_SCOPED_PROFILES
                else DATASETS)
    return tuple(
        FormalSpec(dataset, method, seed)
        for dataset in datasets
        for method in methods
        for seed in seeds
    )


def explanation_specs() -> tuple[FormalSpec, ...]:
    _, _, explanations = _profile_members()
    return tuple(
        FormalSpec(dataset, method, 42, True)
        for dataset in DATASETS
        for method in explanations
    )


def registered_spec_key(spec: FormalSpec) -> str:
    _validate_spec(spec)
    if spec not in set(formal_specs()) | set(explanation_specs()):
        raise ValueError(f'unregistered spec: {spec}')
    return f'{spec.dataset}:{spec.method}:{spec.seed}'


def spec_for_key(key: str) -> FormalSpec:
    if type(key) is not str or not key or '\x00' in key:
        raise ValueError('spec key must be a nonempty string')
    matches = [
        spec for spec in (*formal_specs(), *explanation_specs())
        if registered_spec_key(spec) == key
    ]
    if len(matches) != 1:
        raise ValueError('spec key is not uniquely registered')
    return matches[0]


def safe_spec_name(spec: FormalSpec) -> str:
    name = quote(registered_spec_key(spec), safe='')
    if (not name or name in {'.', '..'} or '/' in name or '\\' in name
            or '..' in name):
        raise ValueError('registered spec has no safe path name')
    return name


def safe_spec_key(key: str) -> str:
    return safe_spec_name(spec_for_key(key))


def protocol_for(spec: FormalSpec) -> dict:
    _validate_spec(spec)
    allowed = set(formal_specs()) | set(explanation_specs())
    if spec not in allowed:
        raise ValueError(f'unregistered spec: {spec}')
    contract = method_contract_for(spec)
    return {
        'dataset': spec.dataset,
        'method': spec.method,
        'seed': spec.seed,
        'explanation': spec.explanation,
        'validation_split_seed': VALIDATION_SPLIT_SEEDS[spec.dataset],
        'er_samples_per_class': contract.get('er_per_class', 0),
        'method_contract': contract,
        'base_options': _base_options(spec),
    }


def validation_access_for(spec):
    protocol_for(spec)
    return spec.method in _VALIDATION_ACCESS_METHODS


def _option_token(option, value):
    kind = OPTION_SCHEMA[option]
    if kind == 'bool':
        return '1' if value else '0'
    if kind == 'text_bool':
        return 'True' if value else 'False'
    if kind == 'tasks':
        return '|'.join(
            ','.join(str(label) for label in task) for task in value)
    return str(value)


def _deployment_value(name, value):
    value = str(value)
    if not value or '\x00' in value:
        raise ValueError(f'{name} must be a non-empty deployment field')
    return value


def _wrapper_for(variant):
    return (
        'from three_dataset_formal_runtime import execute_variant_main; '
        f'execute_variant_main({variant!r})'
    )


def command_for(spec, device, results_dir, smoke=False) -> tuple[str, ...]:
    if type(smoke) is not bool:
        raise TypeError('smoke must be exactly bool')
    if smoke:
        raise ValueError('smoke commands are not part of the reviewed protocol')
    protocol_for(spec)
    device = _deployment_value('device', device)
    results_dir = _deployment_value('results_dir', results_dir)
    options = _without_inherited_method_options(
        _option_map(_base_command(spec, device, results_dir, smoke)))
    root, worktree, python = _deployment_paths()
    options.update(_deployment_option_values(spec, root))
    options['seed'] = str(spec.seed)
    options['formal_deferred_evaluation'] = '1'
    options['exp_name'] = safe_spec_name(spec)
    raw_contract = {**_holdout_contract(spec), **method_contract_for(spec)}
    for option, value in raw_contract.items():
        if option in OPTION_SCHEMA:
            options[option] = _option_token(option, value)
    options['device'] = device
    options['results_dir'] = results_dir
    tokens = tuple(
        token
        for option, value in options.items()
        for token in (f'--{option}', str(value))
    )
    if spec.method in EXTERNAL_METHODS:
        return (str(python), str(worktree / 'main.py'), *tokens)
    return (str(python), '-c', _wrapper_for(spec.method), *tokens)


def parsed_protocol(command, spec):
    expected = protocol_for(spec)
    command = tuple(command)
    if not all(isinstance(token, str) for token in command):
        raise ValueError('command tokens must be strings')
    root, worktree, python = _deployment_paths()
    if spec.method in EXTERNAL_METHODS:
        prefix = (str(python), str(worktree / 'main.py'))
        start = 2
    else:
        prefix = (str(python), '-c', _wrapper_for(spec.method))
        start = 3
    raw_options = _option_map(command, start)
    for option in ('device', 'results_dir'):
        if option not in raw_options:
            raise ValueError(f'command omits deployment option: {option}')
        _deployment_value(option, raw_options[option])
    for option, value in _deployment_option_values(spec, root).items():
        if raw_options.get(option) != value:
            raise ValueError(f'command deployment option differs: {option}')
    if command[:start] != prefix:
        raise ValueError('command entrypoint differs from the frozen protocol')
    normalized = _normalize_options(raw_options)
    contract = method_contract_for(spec)
    if any(
            normalized.get(option) != value
            for option, value in contract.items()
            if option in OPTION_SCHEMA):
        raise ValueError('command method flags differ from the frozen contract')
    scientific = dict(normalized)
    for option in ('data_path', 'vector_npz', 'device', 'results_dir'):
        if option in expected['base_options']:
            scientific[option] = expected['base_options'][option]
    parsed = {
        **{key: value for key, value in expected.items() if key != 'base_options'},
        'base_options': scientific,
    }
    if parsed != expected:
        raise ValueError('command differs from the frozen scientific protocol')
    return parsed


def registry_sha256() -> str:
    payload = [protocol_for(spec) for spec in (*formal_specs(), *explanation_specs())]
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()
