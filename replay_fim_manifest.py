"""Immutable allowlists for the final all-party FIM development attempt."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

from party_drift_manifest import (
    DATASETS, BASELINES, DATA_SHA256, HOLDOUT_SELECTED_SHA256,
    VALIDATION_FILE_SHA256, source_config_path, write_manifest_files,
    _canonical_bytes,
)

SEEDS = (80, 81, 82, 83, 84)
POLICIES = {'f0': 'cumulative', 'f1': 'fresh', 'f_off': 'off'}
ALPHA = 0.25
PROTOCOL = 'allparty-fim-final-attempt-v1'
EPOCHS = {'cifar100': 50, 'isolet': 50, 'upmc_food101': 20}
RESULT_PREFIX = PurePosixPath(
    '/brf/users/yangxx/VFCL-results/allparty-fim-final-attempt-v1-20261029')
SOURCE_COMMIT = 'b2b9546f82765c25e18394874ad0f79e786ffb82'


def expected_job_keys(stage, selected_policy=None):
    if stage == 'A' and selected_policy is None:
        return {
            f'{dataset}/{method}/{"none" if method in BASELINES[dataset] else "0.25"}/{seed}'
            for dataset in DATASETS for seed in SEEDS
            for method in (*POLICIES, *BASELINES[dataset])
        }
    if stage == 'B' and selected_policy in POLICIES:
        return {
            f'{dataset}/no_partykd_{selected_policy}/0.25/{seed}'
            for dataset in DATASETS for seed in SEEDS
        }
    raise ValueError('invalid study stage or selected FIM policy')


def build_job_config(source, *, dataset, method, seed, root, source_config,
                     holdout_manifest, paired_validation_manifest,
                     physical_gpu=None):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError('unlisted dataset or seed')
    policy_key = method.removeprefix('no_partykd_')
    replay = policy_key in POLICIES
    if not replay and method not in BASELINES[dataset]:
        raise ValueError('unlisted study method')
    expected = ('proto_evolve' if replay else
                'der_pp' if method == 'derpp' else method)
    if (source.get('seed') != 42 or source.get('cl_method') != expected or
            source.get('formal_deferred_evaluation') is not True or
            source.get('epochs_per_task') != EPOCHS[dataset] or
            (replay and (source.get('sample_replay_only') != 1 or
                         source.get('fim_freeze_frac') != 0.0 or
                         source.get('party_kd_enabled') != 1 or
                         source.get('party_kd_lambda') != 1.0 or
                         source.get('head_consolidation_samples_per_class') != 20 or
                         source.get('proto_replay_gradient_enabled') != 0 or
                         source.get('replay_mode') != 'prototype'))):
        raise ValueError('source configuration differs from frozen cohort')
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
    alpha_label = '0.25' if replay else 'none'
    job_key = f'{dataset}/{method}/{alpha_label}/{seed}'
    run = root / dataset / method / alpha_label / str(seed)
    data_source = (PurePosixPath(source['data_path']) / 'cifar-100-python' / 'train'
                   if dataset == 'cifar100' else source['vector_npz'])
    physical_gpu = seed % 2 if physical_gpu is None else physical_gpu
    if physical_gpu not in (0, 1):
        raise ValueError('invalid physical GPU assignment')
    config = {
        **source,
        'seed': seed, 'device': 'cuda:0',
        'fim_mask_policy': POLICIES[policy_key] if replay else 'cumulative',
        'party_drift_enabled': int(replay),
        'party_drift_preflight': 0,
        'development_nonfinite_guard': 1,
        'party_replay_arm': 'all_party' if replay else 'off',
        'party_replay_alpha': ALPHA if replay else 1.0,
        'pilot_dataset': dataset,
        'pilot_variant': 'reference20',
        'pilot_protocol': PROTOCOL,
        'pilot_holdout_seed': 20261023,
        'pilot_holdout_per_class': 30,
        'pilot_paired_validation_manifest': str(paired_validation_manifest),
        'development_holdout_manifest': str(holdout_manifest),
        'development_source_config_path': str(source_config),
        'development_source_data_path': str(data_source),
        'development_job_key': job_key,
        'study_physical_gpu': physical_gpu,
        'output_dir': str(run), 'results_dir': str(run.parent),
        'exp_name': run.name,
    }
    if method.startswith('no_partykd_'):
        config.update(party_kd_enabled=0, dep_tracking_enabled=0,
                      expected_party_kd_variant='base')
    return config


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_approved_job(job_key, manifest, digest, *, code_commit, stage,
                      selected_policy=None):
    manifest = Path(manifest).resolve()
    if _sha256(manifest) != digest:
        raise ValueError('study manifest differs from external SHA anchor')
    rows = [json.loads(line) for line in manifest.read_text(
        encoding='utf-8').splitlines()]
    expected = expected_job_keys(stage, selected_policy)
    if (len(rows) != len(expected) or
            {row.get('job_key') for row in rows} != expected or
            any(row.get('code_commit') != code_commit or
                row.get('stage') != stage or
                row.get('source_commit') != SOURCE_COMMIT for row in rows)):
        raise ValueError('study allowlist, stage or code identity differs')
    row = next((item for item in rows if item['job_key'] == job_key), None)
    if row is None:
        raise ValueError('job is not in approved stage')
    config_path = (manifest.parent / row['config_relpath']).resolve()
    if (not config_path.is_relative_to(manifest.parent / 'configs') or
            not config_path.is_file() or
            _sha256(config_path) != row['config_sha256']):
        raise ValueError('approved config path or SHA differs')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dataset, method, alpha, seed = job_key.split('/')
    if (row['dataset'] != dataset or row['method'] != method or
            row['seed'] != int(seed) or
            config.get('development_job_key') != job_key or
            config.get('pilot_protocol') != PROTOCOL or
            config.get('seed') != int(seed) or
            config.get('study_physical_gpu') != row.get('physical_gpu') or
            config.get('fim_mask_policy') != POLICIES.get(
                method.removeprefix('no_partykd_'), 'cumulative') or
            config.get('party_kd_enabled') !=
            (0 if method.startswith('no_partykd_') else
             1 if alpha == '0.25' else config.get('party_kd_enabled')) or
            config.get('party_replay_arm') !=
            ('all_party' if alpha == '0.25' else 'off') or
            (alpha == '0.25' and config.get('party_replay_alpha') != ALPHA)):
        raise ValueError('approved job/config identity differs')
    for field in ('source_config_sha256', 'source_data_sha256',
                  'holdout_sha256', 'validation_manifest_sha256'):
        value = row.get(field)
        if (type(value) is not str or len(value) != 64 or
                any(ch not in '0123456789abcdef' for ch in value)):
            raise ValueError(f'approved {field} is invalid')
    return row


def generate_manifest(*, stage, selected_policy, stage_a_decision,
                      sample_root, baseline_root, evidence_root, result_root,
                      output, code_commit):
    """Write a new, exact-byte stage allowlist after checking frozen inputs."""
    if (len(code_commit) != 40 or
            subprocess.check_output(['git', '-C', str(Path(__file__).parent),
                                     'rev-parse', 'HEAD'], text=True).strip()
            != code_commit):
        raise ValueError('generator code is not at the pinned commit')
    result_root = PurePosixPath(result_root)
    required = RESULT_PREFIX / ('development-a' if stage == 'A' else
                                'development-b')
    if result_root != required:
        raise ValueError('study result root is outside prelocked location')
    required_output = RESULT_PREFIX / 'manifests' / (
        'stage-a' if stage == 'A' else 'stage-b')
    if PurePosixPath(output) != required_output:
        raise ValueError('study manifest output is outside prelocked location')
    decision_sha = None
    if stage == 'B':
        if not stage_a_decision:
            raise ValueError('stage B needs stage-A decision')
        decision = json.loads(Path(stage_a_decision).read_text(encoding='utf-8'))
        if (decision.get('status') != 'selection_passed' or
                decision.get('selected_policy') != selected_policy):
            raise ValueError('stage B selection differs from audited A decision')
        decision_sha = _sha256(stage_a_decision)
    elif stage != 'A' or selected_policy is not None or stage_a_decision:
        raise ValueError('stage A has no selected policy or prior decision')
    sample_root, baseline_root, evidence_root = map(
        Path, (sample_root, baseline_root, evidence_root))
    entries = []
    source_cache, data_cache, holdout_cache, validation_cache = {}, {}, {}, {}
    for index, key in enumerate(sorted(expected_job_keys(stage, selected_policy))):
        dataset, method, _, seed_text = key.split('/')
        replay = method in POLICIES or method.startswith('no_partykd_')
        source_path = Path(source_config_path(
            sample_root, baseline_root, dataset,
            'all_party' if replay else method))
        if source_path not in source_cache:
            source_cache[source_path] = (
                json.loads(source_path.read_text(encoding='utf-8')),
                _sha256(source_path))
        source, source_sha = source_cache[source_path]
        holdout = (evidence_root / 'holdout' /
                   f'{"upmc" if dataset == "upmc_food101" else dataset}'
                   '_tuning_holdout_seed20261023.json')
        if dataset not in holdout_cache:
            locked = json.loads(holdout.read_text(encoding='utf-8'))
            if (locked.get('sha256') != HOLDOUT_SELECTED_SHA256[dataset] or
                    locked.get('source_sha256') != DATA_SHA256[dataset]):
                raise ValueError('holdout differs from prior source anchor')
            holdout_cache[dataset] = _sha256(holdout)
        validation = (sample_root / 'full' / dataset / 'adaptive' /
                      '42' / 'run' / 'validation' / 'validation_manifest.json')
        if dataset not in validation_cache:
            validation_cache[dataset] = _sha256(validation)
            if validation_cache[dataset] != VALIDATION_FILE_SHA256[dataset]:
                raise ValueError('validation differs from prior source anchor')
        if not replay:
            own_validation = (source_path.parent / 'validation' /
                              'validation_manifest.json')
            if _sha256(own_validation) != validation_cache[dataset]:
                raise ValueError('baseline validation split differs')
        config = build_job_config(
            source, dataset=dataset, method=method, seed=int(seed_text),
            root=result_root, source_config=source_path,
            holdout_manifest=holdout,
            paired_validation_manifest=validation, physical_gpu=index % 2)
        data_source = Path(config['development_source_data_path'])
        if data_source not in data_cache:
            data_cache[data_source] = _sha256(data_source)
        if data_cache[data_source] != DATA_SHA256[dataset]:
            raise ValueError('training data differs from prior source anchor')
        row = {
            'job_key': key, 'dataset': dataset, 'method': method,
            'seed': int(seed_text), 'stage': stage,
            'physical_gpu': index % 2,
            'code_commit': code_commit, 'source_commit': SOURCE_COMMIT,
            'config_relpath': 'configs/' + key.replace('/', '_').replace('.', 'p') + '.json',
            'config_sha256': hashlib.sha256(_canonical_bytes(config)).hexdigest(),
            'source_config_sha256': source_sha,
            'source_data_sha256': data_cache[data_source],
            'holdout_sha256': holdout_cache[dataset],
            'validation_manifest_sha256': validation_cache[dataset],
            'stage_a_decision_sha256': decision_sha,
        }
        entries.append((row, config))
    if len(entries) != len(expected_job_keys(stage, selected_policy)):
        raise AssertionError('study matrix is incomplete')
    return write_manifest_files(output, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('A', 'B'), required=True)
    parser.add_argument('--selected-policy', choices=tuple(POLICIES))
    parser.add_argument('--stage-a-decision', type=Path)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    print(generate_manifest(**vars(args)))


if __name__ == '__main__':
    main()
