"""Build exact one-attempt configs for frozen-method formal module ablations."""

import argparse
import hashlib
import json
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from pathlib import PurePosixPath

SOURCE_ROOT = Path('/brf/users/yangxx/VFCL-results/adaptive-sample-only-fourparty-v1-20261017')
RESULT_ROOT = PurePosixPath(
    '/brf/users/yangxx/VFCL-results/frozen-method-module-ablation-v2r2-20261101')
SMOKE_PRODUCER_COMMIT = 'af049e37e37fd2882bb8768913e55be96c0a50c3'
BASE_COMMIT = 'e2a559dc4945729848225cf21fdc95451762c231'
PATCH_COMMIT = 'b13ac243d7b4acd3f6a512e4addc7122e4f4a38c'
SOURCE_ANCHOR_SHA = '08db7e2c0c2ee6c1471aa576812baa3de6a1f95268b582d53df7a57ed66d8f3c'
DATA_SHA = {
    'cifar100': '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    'isolet': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'upmc_food101': 'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
}
AUX_DATA_SHA = {
    'cifar100': {
        'meta': 'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9',
        'test': '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864',
    },
    'isolet': {
        'metadata': '79396dea1751b6094a5769f2dd789ad58b3ea9582a12c8c98d3ec07d8d1eb3cd',
    },
    'upmc_food101': {
        'metadata': 'a9f81a847969dab024d0831e6505e764f8fce58d27828f18eb593d8001358c8c',
    },
}
DATASETS = ('cifar100', 'isolet', 'upmc_food101')
SEEDS = (42, 43, 44)
ARMS = ('no_partykd', 'fim_off')
EPOCHS = {'cifar100': 50, 'isolet': 50, 'upmc_food101': 20}
RUNTIME_FIELDS = frozenset(('output_dir', 'results_dir', 'exp_name'))
PREFLIGHT_FLAGS = {
    'default_on_bitwise': True,
    'off_task0_identical': True,
    'sample_order_identical_all_tasks': True,
    'fim_tensor_values_compared': True,
    'off_task1_model_diverged': True,
    'off_applied_freeze_fraction': 0,
    'test_accessed': False,
}


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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def patch_diff_sha256():
    diff = subprocess.check_output([
        'git', '-C', str(Path(__file__).parent), 'diff',
        f'{BASE_COMMIT}..{PATCH_COMMIT}', '--',
        'config.py', 'cl_methods/proto_evolve.py',
    ])
    return hashlib.sha256(diff).hexdigest()


def validate_smoke_comparison(comparison):
    if not isinstance(comparison, dict) or set(comparison) != set(DATASETS):
        raise ValueError('preflight smoke comparison must cover all datasets')
    for dataset, report in comparison.items():
        if not isinstance(report, dict) or any(
                report.get(field) != value
                for field, value in PREFLIGHT_FLAGS.items()):
            raise ValueError(f'preflight smoke did not pass for {dataset}')
        for field in ('raw_smoke_sha256', 'raw_config_sha256',
                      'raw_no_test_sha256'):
            hashes = report.get(field)
            if (not isinstance(hashes, dict) or
                    set(hashes) != {'base', 'default_on', 'off'} or
                    any(not isinstance(value, str) or len(value) != 64
                        for value in hashes.values())):
                raise ValueError(f'preflight raw smoke hashes are missing: {dataset}')
        if (not isinstance(report.get('orchestration_commit'), str) or
                len(report['orchestration_commit']) != 40 or
                not isinstance(report.get('smoke_script_sha256'), str) or
                len(report['smoke_script_sha256']) != 64):
            raise ValueError(f'preflight smoke code identity is missing: {dataset}')


def _pretty_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) +
            '\n').encode('utf-8')


def runtime_identity():
    import numpy as np
    import PIL
    import sklearn
    import torch
    import torchvision
    return {
        'cuda_runtime': torch.version.cuda,
        'host': socket.gethostname(),
        'numpy': np.__version__,
        'pillow': PIL.__version__,
        'python': platform.python_version(),
        'scikit_learn': sklearn.__version__,
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
    }


def validate_publication_preflight(root):
    root = Path(root)
    report_path = root / 'PUBLICATION_PREFLIGHT.json'
    report = json.loads(report_path.read_text())
    log_text = (root / 'publication-tests.log').read_text(encoding='utf-8')
    code = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if (report.get('status') != 'passed' or report.get('returncode') != 0 or
            report.get('platform') != 'linux' or
            report.get('linux_producer_fixture_passed') is not True or
            report.get('skipped_tests') != 0 or
            'skipped' in log_text.lower() or
            not re.search(r'^test_full_linux_producer_fixture_passes_formal_seal .* \.\.\. ok$', log_text, re.M) or
            report.get('orchestration_commit') != commit or
            report.get('real_test_access') is not False or
            report.get('log_sha256') != _sha(root / 'publication-tests.log')):
        raise ValueError('synthetic publication preflight did not pass this code')
    modules = ('test_frozen_ablation_run', 'test_frozen_ablation_audit',
               'test_frozen_ablation_manifest', 'test_frozen_ablation_smoke',
               'test_frozen_fim_application')
    if report.get('test_source_sha256') != {m: _sha(code / (m + '.py')) for m in modules}:
        raise ValueError('publication preflight test sources changed')
    return _sha(report_path)


def load_preflight(smoke_path, smoke_root, review_path, publication_root):
    smoke_bytes = Path(smoke_path).read_bytes()
    review_bytes = Path(review_path).read_bytes()
    comparison = json.loads(smoke_bytes)
    review = json.loads(review_bytes)
    validate_smoke_comparison(comparison)
    from frozen_ablation_smoke import validate_smoke_records
    validate_smoke_records(smoke_root, comparison,
                           producer_commit=SMOKE_PRODUCER_COMMIT)
    publication_sha = validate_publication_preflight(publication_root)
    expected_diff = patch_diff_sha256()
    if (review.get('verdict') != 'approved' or
            not isinstance(review.get('reviewer_id'), str) or
            not review['reviewer_id'].strip() or
            review.get('base_commit') != BASE_COMMIT or
            review.get('patch_commit') != PATCH_COMMIT or
            review.get('patch_diff_sha256') != expected_diff or
            review.get('blocking_findings') != []):
        raise ValueError('independent patch review is absent or does not approve this diff')
    record = {
        'schema_version': 1,
        'status': 'approved',
        'reviewer_id': review['reviewer_id'],
        'base_commit': BASE_COMMIT,
        'patch_commit': PATCH_COMMIT,
        'patch_diff_sha256': expected_diff,
        'smoke_comparison_sha256': hashlib.sha256(smoke_bytes).hexdigest(),
        'patch_review_sha256': hashlib.sha256(review_bytes).hexdigest(),
        'publication_preflight_sha256': publication_sha,
    }
    preflight_bytes = _pretty_bytes(record)
    record['preflight_sha256'] = hashlib.sha256(preflight_bytes).hexdigest()
    smoke_root = Path(smoke_root)
    smoke_files = {}
    for dataset in DATASETS:
        for mode in ('base', 'default_on', 'off'):
            for name in ('SMOKE_RESULT.json', 'SMOKE_CONFIG.json',
                         'NO_TEST_ACCESS.json'):
                relative = Path('smokes') / dataset / mode / name
                smoke_files[relative] = (
                    smoke_root / dataset / mode / name).read_bytes()
    for name in ('PUBLICATION_PREFLIGHT.json', 'publication-tests.log'):
        smoke_files[Path('publication') / name] = (Path(publication_root) / name).read_bytes()
    return {'record': record, 'preflight_bytes': preflight_bytes,
            'smoke_bytes': smoke_bytes, 'review_bytes': review_bytes,
            'smoke_files': smoke_files}


def original_tree_sha256(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('original formal job tree is absent or a symlink')
    rows = []
    for path in sorted(root.rglob('*'),
                       key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError('original formal job tree contains a symlink')
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(),
                         path.stat().st_size, _sha(path)])
    return hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def auxiliary_data_paths(source, dataset):
    if dataset == 'cifar100':
        base = Path(source['data_path']) / 'cifar-100-python'
        return {name: base / name for name in AUX_DATA_SHA[dataset]}
    base = Path(source['vector_npz'])
    return {'metadata': base.with_suffix('.metadata.json')}


def manifest_entries(preflight):
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
    gate_path = SOURCE_ROOT / anchor['final_gate']['relative_path']
    if _sha(gate_path) != anchor['final_gate']['sha256']:
        raise ValueError('original formal final gate changed')
    final_gate = json.loads(gate_path.read_text(encoding='utf-8'))
    data_cache, tree_cache = {}, {}
    entries = []
    current_runtime = runtime_identity()
    uuids = {0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
             1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1'}
    for key in sorted(expected_keys()):
        dataset, arm, seed_text = key.split('/')
        seed = int(seed_text)
        source_path = (SOURCE_ROOT / 'full' / dataset / 'adaptive' /
                       str(seed) / 'run/config.json')
        source_job = source_path.parent.parent
        old_key = f'{dataset}/adaptive/{seed}'
        if source_job not in tree_cache:
            tree_cache[source_job] = original_tree_sha256(source_job)
        tree_sha = tree_cache[source_job]
        if final_gate['original_tree_sha256'].get(old_key) != tree_sha:
            raise ValueError(f'original formal job tree changed: {old_key}')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        original_success = json.loads((source_job / 'JOB_SUCCESS.json').read_text())
        formal_runtime = original_success.get('runtime')
        if not isinstance(formal_runtime, dict) or current_runtime != formal_runtime:
            raise ValueError(f'current runtime differs from original formal run: {key}')
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
        for name, path in auxiliary_data_paths(source, dataset).items():
            if path not in data_cache:
                data_cache[path] = _sha(path)
            if data_cache[path] != AUX_DATA_SHA[dataset][name]:
                raise ValueError(f'formal auxiliary data SHA differs: {key}, {name}')
        config_relpath = 'configs/' + key.replace('/', '_') + '.json'
        row = {
            'job_key': key, 'dataset': dataset, 'arm': arm, 'seed': seed,
            'physical_gpu': gpu, 'gpu_uuid': uuids[gpu],
            'code_commit': BASE_COMMIT if arm == 'no_partykd' else PATCH_COMMIT,
            'orchestration_commit': orchestration_commit,
            'source_commit': BASE_COMMIT,
            'source_formal_anchor_sha256': SOURCE_ANCHOR_SHA,
            'source_config_sha256': _sha(source_path),
            'original_tree_sha256': tree_sha,
            'runtime': formal_runtime,
            'runtime_sha256': hashlib.sha256(_bytes(formal_runtime)).hexdigest(),
            'source_data_sha256': data_cache[data],
            'auxiliary_data_sha256': AUX_DATA_SHA[dataset],
            'config_relpath': config_relpath,
            'config_sha256': hashlib.sha256(_bytes(config)).hexdigest(),
            'config_semantic_diff': semantic_diff(source, config),
            'preflight_sha256': preflight['record']['preflight_sha256'],
            'smoke_comparison_sha256': preflight['record'][
                'smoke_comparison_sha256'],
            'patch_review_sha256': preflight['record']['patch_review_sha256'],
            'patch_diff_sha256': preflight['record']['patch_diff_sha256'],
            'publication_preflight_sha256': preflight['record']['publication_preflight_sha256'],
        }
        entries.append((row, config))
    if len(entries) != 18:
        raise AssertionError('formal ablation matrix is incomplete')
    return entries


def write_manifest(entries, preflight):
    root = Path(RESULT_ROOT)
    if root.exists():
        raise FileExistsError('ablation result root already exists')
    if shutil.disk_usage(root.parent).free < 40 * (1 << 30):
        raise RuntimeError('formal ablation needs at least 40 GiB free before manifest write')
    manifest_dir = root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = manifest_dir / 'preflight'
    evidence_dir.mkdir()
    for name, data in (
            ('PRELAUNCH_PREFLIGHT.json', preflight['preflight_bytes']),
            ('SMOKE_COMPARISON.json', preflight['smoke_bytes']),
            ('PATCH_REVIEW.json', preflight['review_bytes'])):
        with (evidence_dir / name).open('xb') as stream:
            stream.write(data)
    for relative, data in preflight['smoke_files'].items():
        path = evidence_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
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
    parser.add_argument('--smoke-comparison', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--patch-review', type=Path, required=True)
    parser.add_argument('--publication-preflight', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
        text=True).strip()
    status = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parent), 'status', '--porcelain'],
        text=True).strip()
    if status:
        raise ValueError('manifest generator checkout is not clean')
    preflight = load_preflight(args.smoke_comparison, args.smoke_root,
                               args.patch_review, args.publication_preflight)
    entries = manifest_entries(preflight)
    payload = b''.join(_bytes(row) for row, _ in entries)
    preview = {'jobs': len(entries), 'manifest_sha256': hashlib.sha256(payload).hexdigest(),
               'orchestration_commit': commit, 'written': False}
    if args.write:
        if write_manifest(entries, preflight) != preview['manifest_sha256']:
            raise AssertionError('written manifest differs from dry run')
        preview['written'] = True
    print(json.dumps(preview, sort_keys=True))


if __name__ == '__main__':
    main()
