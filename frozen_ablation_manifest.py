"""Build exact one-attempt configs for frozen-method formal module ablations."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from pathlib import PurePosixPath

SOURCE_ROOT = Path('/brf/users/yangxx/VFCL-results/adaptive-sample-only-fourparty-v1-20261017')
RESULT_ROOT = PurePosixPath(
    '/brf/users/yangxx/VFCL-results/frozen-method-module-ablation-v1-20261031')
BASE_COMMIT = 'e2a559dc4945729848225cf21fdc95451762c231'
PATCH_COMMIT = 'b13ac243d7b4acd3f6a512e4addc7122e4f4a38c'
SOURCE_ANCHOR_SHA = '08db7e2c0c2ee6c1471aa576812baa3de6a1f95268b582d53df7a57ed66d8f3c'
DATA_SHA = {
    'cifar100': '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    'isolet': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'upmc_food101': 'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
}
DATASETS = ('cifar100', 'isolet', 'upmc_food101')
SEEDS = (42, 43, 44)
ARMS = ('no_partykd', 'fim_off')
EPOCHS = {'cifar100': 50, 'isolet': 50, 'upmc_food101': 20}
RUNTIME_FIELDS = frozenset(('output_dir', 'results_dir', 'exp_name'))


def expected_keys():
    return {f'{dataset}/{arm}/{seed}'
            for dataset in DATASETS for arm in ARMS for seed in SEEDS}


def semantic_diff(source, target):
    return {key: [source.get(key), target.get(key)]
            for key in sorted(set(source) | set(target))
            if key not in RUNTIME_FIELDS and source.get(key) != target.get(key)}


def derive_config(source, dataset, arm, seed):
    if (dataset not in DATASETS or arm not in ARMS or seed not in SEEDS or
            source.get('seed') != seed or
            source.get('cl_method') != 'proto_evolve' or
            source.get('formal_deferred_evaluation') is not True or
            source.get('sample_replay_only') != 1 or
            source.get('head_consolidation_samples_per_class') != 20 or
            source.get('epochs_per_task') != EPOCHS[dataset] or
            source.get('party_kd_enabled') != 1 or
            source.get('dep_tracking_enabled') != 1 or
            source.get('expected_party_kd_variant') != 'uniform' or
            source.get('fim_freeze_frac') != 0.0 or
            'fim_apply_mask' in source):
        raise ValueError('source differs from the frozen formal method')
    target = dict(source)
    if arm == 'no_partykd':
        target.update(party_kd_enabled=0, dep_tracking_enabled=0,
                      expected_party_kd_variant='base')
    else:
        target['fim_apply_mask'] = 0
    run = RESULT_ROOT / arm / dataset / str(seed) / 'attempt-01'
    target.update(results_dir=str(run.parent), output_dir=str(run),
                  exp_name=run.name)
    expected = ({'party_kd_enabled', 'dep_tracking_enabled',
                 'expected_party_kd_variant'} if arm == 'no_partykd'
                else {'fim_apply_mask'})
    if set(semantic_diff(source, target)) != expected:
        raise AssertionError('derived config has an unapproved semantic difference')
    return target


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False) + '\n').encode('utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_entries():
    orchestration_commit = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
        text=True).strip()
    anchor_path = SOURCE_ROOT / 'audit/ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json'
    if _sha(anchor_path) != SOURCE_ANCHOR_SHA:
        raise ValueError('original formal anchor changed')
    anchor = json.loads(anchor_path.read_text(encoding='utf-8'))
    if (anchor.get('producer_commit') != BASE_COMMIT or
            anchor.get('basic_passed') != 9 or
            anchor.get('verified_original_trees') != 9):
        raise ValueError('original nine formal jobs are not anchored')
    data_cache = {}
    entries = []
    uuids = {0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
             1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1'}
    for key in sorted(expected_keys()):
        dataset, arm, seed_text = key.split('/')
        seed = int(seed_text)
        source_path = (SOURCE_ROOT / 'full' / dataset / 'adaptive' /
                       str(seed) / 'run/config.json')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        if (source.get('output_dir') != str(source_path.parent) or
                source.get('results_dir') != str(source_path.parent.parent) or
                source.get('exp_name') != 'run'):
            raise ValueError(f'formal source path differs: {key}')
        config = derive_config(source, dataset, arm, seed)
        device = config.get('device')
        if device not in ('cuda:0', 'cuda:1'):
            raise ValueError(f'formal GPU assignment differs: {key}')
        gpu = int(device[-1])
        data = (Path(source['data_path']) / 'cifar-100-python/train'
                if dataset == 'cifar100' else Path(source['vector_npz']))
        if data not in data_cache:
            data_cache[data] = _sha(data)
        if data_cache[data] != DATA_SHA[dataset]:
            raise ValueError(f'training data SHA differs: {key}')
        config_relpath = 'configs/' + key.replace('/', '_') + '.json'
        row = {
            'job_key': key, 'dataset': dataset, 'arm': arm, 'seed': seed,
            'physical_gpu': gpu, 'gpu_uuid': uuids[gpu],
            'code_commit': BASE_COMMIT if arm == 'no_partykd' else PATCH_COMMIT,
            'orchestration_commit': orchestration_commit,
            'source_commit': BASE_COMMIT,
            'source_formal_anchor_sha256': SOURCE_ANCHOR_SHA,
            'source_config_sha256': _sha(source_path),
            'source_data_sha256': data_cache[data],
            'config_relpath': config_relpath,
            'config_sha256': hashlib.sha256(_bytes(config)).hexdigest(),
            'config_semantic_diff': semantic_diff(source, config),
        }
        entries.append((row, config))
    if len(entries) != 18:
        raise AssertionError('formal ablation matrix is incomplete')
    return entries


def write_manifest(entries):
    root = Path(RESULT_ROOT)
    if root.exists():
        raise FileExistsError('ablation result root already exists')
    manifest_dir = root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=False)
    for row, config in entries:
        path = manifest_dir / row['config_relpath']
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:
            stream.write(_bytes(config))
        if _sha(path) != row['config_sha256']:
            raise AssertionError('config SHA changed during manifest write')
    payload = b''.join(_bytes(row) for row, _ in entries)
    with (manifest_dir / 'manifest.jsonl').open('xb') as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with (manifest_dir / 'manifest.sha256').open('x', encoding='ascii') as stream:
        stream.write(digest + '\n')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    commit = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
        text=True).strip()
    status = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parent), 'status', '--porcelain'],
        text=True).strip()
    if status:
        raise ValueError('manifest generator checkout is not clean')
    entries = manifest_entries()
    payload = b''.join(_bytes(row) for row, _ in entries)
    preview = {'jobs': len(entries), 'manifest_sha256': hashlib.sha256(payload).hexdigest(),
               'orchestration_commit': commit, 'written': False}
    if args.write:
        if write_manifest(entries) != preview['manifest_sha256']:
            raise AssertionError('written manifest differs from dry run')
        preview['written'] = True
    print(json.dumps(preview, sort_keys=True))


if __name__ == '__main__':
    main()
