"""Sealed nine-job allowlist for the disclosed F-off formal follow-up."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

from party_drift_manifest import DATASETS, DATA_SHA256, _canonical_bytes, write_manifest_files
from replay_fim_manifest import EPOCHS

ROOT = PurePosixPath('/brf/users/yangxx/VFCL-results/f-off-formal-followup-v3-20261102')
SOURCE_ROOT = PurePosixPath('/brf/users/yangxx/VFCL-results/adaptive-sample-only-fourparty-v1-20261017')
SOURCE_CONFIG_SHA256 = {
    'cifar100': {
        42: '673ca34abe87179a75abff9d10d7e7a7392f786587df8c4d668340b57418a8c2',
        43: '701366bb9ecadf3012c686b5f6db95a667a832ef3bcffee3bc7c0fb83c710299',
        44: 'd6a6ff59066077128346b28d76dbd21a5f5e37c71883c1d75fec248d1da135a5',
    },
    'isolet': {
        42: '9a524718f29e3dbae0a0157a875ce2b4eccb27aa51db7f757dd00858240bc8db',
        43: '2ddc76bfc50930be1e6efe5f8f18897d6ca8b96b8b5a1f5d995ea9a8733cbeaa',
        44: '5f949a01504ffab63434596bb94bd9321f177539b3a50156fffe2e987b6beb68',
    },
    'upmc_food101': {
        42: '4e8c76695ad2c8cd4d2ddd895e134b386597281a756d80c2c3555ffb5a9042e1',
        43: 'da8b57a47657e6ffcaf6786066eaad21f76cee8d4a67d85135c44be1f46b3428',
        44: 'cedd863562bdb92e38a02d64abb441638f7ed649e4ab6f15ec747eb5c63b5be3',
    },
}
SEEDS = (42, 43, 44)
CODE_SOURCE_COMMIT = 'b2b9546f82765c25e18394874ad0f79e786ffb82'
STAGE_A_SHA = '46a1081417d262f9de737f32a229705f1b46469f451813bcabaee2f7f2aa36e2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_checkout(commit):
    checkout = Path(__file__).resolve().parent
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                   text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status',
                                     '--porcelain', '--untracked-files=all'],
                                    text=True).strip()
    if head != commit or len(commit) != 40 or dirty:
        raise ValueError('executed checkout differs from clean pinned commit')
    return commit


def expected_keys():
    return {f'{dataset}/f_off/0.25/{seed}'
            for dataset in DATASETS for seed in SEEDS}


def validate_stage_a(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError('stage-A decision differs from the recorded closure')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (value.get('stage') != 'A' or value.get('status') != 'direction_closed'
            or value.get('selected_policy') is not None
            or value.get('official_test_accessed') is not False):
        raise ValueError('formal follow-up must retain the closed stage-A history')
    return expected_sha


def build_config(source, dataset, seed, physical_gpu):
    if (dataset not in DATASETS or seed not in SEEDS or physical_gpu not in (0, 1)
            or source.get('seed') != seed
            or source.get('cl_method') != 'proto_evolve'
            or source.get('formal_deferred_evaluation') is not True
            or source.get('epochs_per_task') != EPOCHS[dataset]
            or source.get('num_parties') != 4
            or source.get('sample_replay_only') != 1
            or source.get('fim_freeze_frac') != 0.0
            or source.get('party_kd_enabled') != 1
            or source.get('party_kd_lambda') != 1.0
            or source.get('expected_party_kd_variant') != 'uniform'
            or source.get('head_consolidation_samples_per_class') != 20
            or source.get('proto_replay_gradient_enabled') != 0
            or source.get('replay_mode') != 'prototype'):
        raise ValueError('source config differs from locked formal protocol')
    key = f'{dataset}/f_off/0.25/{seed}'
    run = ROOT / 'formal' / key
    config = {name: value for name, value in source.items()
              if not name.startswith('development_')
              and name != 'formal_source_provenance'}
    config.update({
        'device': 'cuda:0', 'fim_mask_policy': 'off',
        'party_drift_enabled': 1, 'party_drift_preflight': 0,
        'development_nonfinite_guard': 1,
        'party_replay_arm': 'all_party', 'party_replay_alpha': 0.25,
        'study_physical_gpu': physical_gpu,
        'formal_followup_job_key': key,
        'output_dir': str(run), 'results_dir': str(run.parent),
        'exp_name': run.name,
    })
    return config


def generate_manifest(sample_root, stage_a_decision, output, code_commit):
    pinned_checkout(code_commit)
    if PurePosixPath(output) != ROOT / 'manifests/formal':
        raise ValueError('formal manifest output differs from new protocol root')
    validate_stage_a(stage_a_decision, STAGE_A_SHA)
    if PurePosixPath(sample_root) != SOURCE_ROOT:
        raise ValueError('formal source root differs from original frozen cohort')
    sample_root = Path(sample_root)
    entries = []
    for index, key in enumerate(sorted(expected_keys())):
        dataset, _, _, seed_text = key.split('/')
        seed = int(seed_text)
        source_path = sample_root / 'full' / dataset / 'adaptive' / seed_text / 'run/config.json'
        if digest(source_path) != SOURCE_CONFIG_SHA256[dataset][seed]:
            raise ValueError(f'formal source config differs: {dataset}/{seed}')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        config = build_config(source, dataset, seed, index % 2)
        data_path = (Path(source['data_path']) / 'cifar-100-python/train'
                     if dataset == 'cifar100' else Path(source['vector_npz']))
        data_sha = digest(data_path)
        if data_sha != DATA_SHA256[dataset]:
            raise ValueError(f'formal source data differs: {dataset}')
        validation = source_path.parent / 'validation/validation_manifest.json'
        row = {
            'stage': 'formal_followup', 'job_key': key, 'dataset': dataset,
            'method': 'f_off', 'seed': seed, 'physical_gpu': index % 2,
            'code_commit': code_commit, 'source_commit': CODE_SOURCE_COMMIT,
            'stage_a_decision_sha256': STAGE_A_SHA,
            'source_config_path': str(source_path),
            'source_config_sha256': SOURCE_CONFIG_SHA256[dataset][seed],
            'source_data_path': str(data_path), 'source_data_sha256': data_sha,
            'validation_manifest_path': str(validation),
            'validation_manifest_sha256': digest(validation),
            'config_relpath': 'configs/' + key.replace('/', '_').replace('.', 'p') + '.json',
            'config_sha256': hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        }
        entries.append((row, config))
    return write_manifest_files(output, entries)


def load_approved_job(job_key, manifest, manifest_sha, code_commit):
    manifest = Path(manifest).resolve()
    if digest(manifest) != manifest_sha:
        raise ValueError('formal manifest differs from sealed SHA')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if (len(rows) != 9 or {row.get('job_key') for row in rows} != expected_keys()
            or any(row.get('dataset') not in DATASETS
                   or row.get('seed') not in SEEDS
                   or row.get('stage') != 'formal_followup'
                   or row.get('code_commit') != code_commit
                   or row.get('stage_a_decision_sha256') != STAGE_A_SHA
                   or row.get('source_config_path') != str(
                       SOURCE_ROOT / 'full' / row['dataset'] / 'adaptive'
                       / str(row['seed']) / 'run/config.json')
                   or row.get('source_config_sha256') !=
                       SOURCE_CONFIG_SHA256[row['dataset']][row['seed']]
                   for row in rows)):
        raise ValueError('formal allowlist or code identity differs')
    if job_key not in expected_keys():
        raise ValueError('formal job is outside approved allowlist')
    row = next(item for item in rows if item['job_key'] == job_key)
    path = (manifest.parent / row['config_relpath']).resolve()
    if (not path.is_relative_to((manifest.parent / 'configs').resolve())
            or digest(path) != row['config_sha256']):
        raise ValueError('formal config path or SHA differs')
    config = json.loads(path.read_text(encoding='utf-8'))
    dataset, _, _, seed = job_key.split('/')
    if (row['dataset'] != dataset or row['seed'] != int(seed)
            or config.get('formal_followup_job_key') != job_key
            or config.get('output_dir') != str(ROOT / 'formal' / job_key)
            or config.get('study_physical_gpu') != row.get('physical_gpu')
            or config.get('fim_mask_policy') != 'off'
            or config.get('party_replay_arm') != 'all_party'
            or config.get('party_replay_alpha') != 0.25):
        raise ValueError('formal job/config identity differs')
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--stage-a-decision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    print(generate_manifest(**vars(parser.parse_args())))
