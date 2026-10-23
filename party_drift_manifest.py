"""Immutable development-job allowlist for the preregistered 183 runs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
SEEDS = (70, 71, 72)
ALPHAS = (0.25, 0.5, 1.0)
REPLAY_ARMS = ('all_party', 'uniform', 'static', 'drift',
               'cmcdr', 'cmcdr_budget')
BASELINES = {
    'cifar100': ('er_ace',),
    'isolet': ('icarl',),
    'upmc_food101': ('derpp', 'icarl'),
}
DATA_SHA256 = {
    'isolet': 'd34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514',
    'upmc_food101': 'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b',
    'cifar100': '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
}
HOLDOUT_SELECTED_SHA256 = {
    'isolet': '668cf1ce874c55a587647940d58181d7a42dc0ae0aa75c1dd4625004ef923089',
    'upmc_food101': 'd19918d00acf93cd65f3095ea3a61914da54144e192ee0fd411d76bfaf3e595f',
    'cifar100': '55b8130fee6f828d0e6be8fbc39e8db70f05aebb55130073da47873c77a9e5ea',
}
VALIDATION_FILE_SHA256 = {
    'isolet': 'e4fc6f8f6a0e960cd36f8904046449f4f2e023e59c42e11edbbf203e0df8d9e2',
    'upmc_food101': 'a6d9105a9116d0ee8638fa0d8c6cf1379945f743e5d0e67269823017327f10f2',
    'cifar100': '7934fcbc5385883cd4312d7416d70dbae8a73044802431f766ee6de2249e5418',
}


def source_config_path(sample_root, baseline_root, dataset, method):
    if dataset not in DATASETS:
        raise ValueError('unknown development dataset')
    if method in REPLAY_ARMS or method == 'sample_only_off':
        root = sample_root
        parts = ('full', dataset, 'adaptive', '42', 'run', 'config.json')
    elif method in BASELINES[dataset]:
        root = baseline_root
        source_method = 'der_pp' if method == 'derpp' else method
        parts = ('full-3080', dataset, source_method, '42', 'run', 'config.json')
    else:
        raise ValueError('unlisted development method')
    root = PurePosixPath(root) if str(root).startswith('/') else Path(root)
    return root.joinpath(*parts)


def build_job_config(source, *, dataset, method, alpha, seed, root,
                     source_config, holdout_manifest,
                     paired_validation_manifest):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError('unlisted development dataset or seed')
    replay = method in REPLAY_ARMS
    if replay and alpha not in ALPHAS or not replay and alpha is not None:
        raise ValueError('unlisted replay alpha')
    if not replay and method not in BASELINES[dataset] + ('sample_only_off',):
        raise ValueError('unlisted development method')
    expected = (
        'proto_evolve' if replay or method == 'sample_only_off'
        else 'der_pp' if method == 'derpp' else method
    )
    if (source.get('seed') != 42 or source.get('cl_method') != expected or
            (expected == 'proto_evolve' and source.get('sample_replay_only') != 1)
            or source.get('formal_deferred_evaluation') is not True):
        raise ValueError('wrong frozen source configuration for method')
    root = (PurePosixPath(root) if str(root).startswith('/')
            else Path(root).resolve())
    alpha_label = 'none' if alpha is None else str(alpha)
    key = f'{dataset}/{method}/{alpha_label}/{seed}'
    run = root / dataset / method / alpha_label / str(seed)
    data_source = (PurePosixPath(source['data_path']) / 'cifar-100-python' / 'train'
                   if dataset == 'cifar100' else source['vector_npz'])
    return {
        **source,
        'seed': seed,
        'device': 'cuda:0' if seed % 2 == 0 else 'cuda:1',
        'party_drift_enabled': int(replay),
        'party_drift_preflight': 0,
        'development_nonfinite_guard': 1,
        'party_replay_arm': method if replay else 'off',
        'party_replay_alpha': float(alpha) if replay else 1.0,
        'pilot_dataset': dataset,
        'pilot_variant': 'reference20',
        'pilot_protocol': 'party-drift-development-v1',
        'pilot_holdout_seed': 20261023,
        'pilot_holdout_per_class': 30,
        'pilot_paired_validation_manifest': str(paired_validation_manifest),
        'development_holdout_manifest': str(holdout_manifest),
        'development_source_config_path': str(source_config),
        'development_source_data_path': str(data_source),
        'development_job_key': key,
        'output_dir': str(run),
        'results_dir': str(run.parent),
        'exp_name': run.name,
    }


def expected_job_keys():
    keys = set()
    for dataset in DATASETS:
        for seed in SEEDS:
            keys.add(f'{dataset}/sample_only_off/none/{seed}')
            for baseline in BASELINES[dataset]:
                keys.add(f'{dataset}/{baseline}/none/{seed}')
            for arm in REPLAY_ARMS:
                for alpha in ALPHAS:
                    keys.add(f'{dataset}/{arm}/{alpha}/{seed}')
    if len(keys) != 183:
        raise AssertionError('development matrix is not 183 unique jobs')
    return keys


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def write_manifest_files(output, entries):
    """Write new config files and their exact-byte canonical allowlist once."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row, config in entries:
        target = (output / row['config_relpath']).resolve()
        if not target.is_relative_to(output.resolve() / 'configs'):
            raise ValueError('config escaped manifest directory')
        content = _canonical_bytes(config)
        if hashlib.sha256(content).hexdigest() != row['config_sha256']:
            raise ValueError('config SHA-256 differs before manifest write')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        rows.append(row)
    content = b''.join(_canonical_bytes(row)
                       for row in sorted(rows, key=lambda item: item['job_key']))
    digest = hashlib.sha256(content).hexdigest()
    with (output / 'manifest.jsonl').open('xb') as stream:
        stream.write(content)
    with (output / 'manifest.sha256').open('x', encoding='ascii') as stream:
        stream.write(digest + '\n')
    return digest


def load_approved_job(job_key, manifest, digest, *, production_commit):
    manifest = Path(manifest).resolve()
    if _sha256(manifest) != digest:
        raise ValueError('manifest SHA-256 differs from external anchor')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if (len(rows) != 183 or {row.get('job_key') for row in rows} !=
            expected_job_keys()):
        raise ValueError('development manifest has missing or duplicate jobs')
    if any(row.get('production_commit') != production_commit for row in rows):
        raise ValueError('manifest production commit differs')
    selected = next((row for row in rows if row['job_key'] == job_key), None)
    if selected is None:
        raise ValueError('job is outside the approved development manifest')
    dataset, method, alpha, seed = job_key.split('/')
    if (selected.get('dataset') != dataset or selected.get('method') != method or
            selected.get('seed') != int(seed) or selected.get('alpha') !=
            (None if alpha == 'none' else float(alpha))):
        raise ValueError('manifest row identity differs from job key')
    path = (manifest.parent / selected['config_relpath']).resolve()
    if (not path.is_relative_to(manifest.parent / 'configs') or
            not path.is_file() or _sha256(path) != selected['config_sha256']):
        raise ValueError('approved job config path or SHA-256 differs')
    config = json.loads(path.read_text(encoding='utf-8'))
    if (config.get('development_job_key') != job_key or
            config.get('seed') != int(seed) or
            config.get('pilot_dataset') != dataset or
            (alpha != 'none' and (
                config.get('party_replay_arm') != method or
                config.get('party_replay_alpha') != float(alpha)))):
        raise ValueError('approved config contents differ from manifest row')
    for field in ('source_config_sha256', 'source_data_sha256',
                  'holdout_sha256', 'validation_manifest_sha256'):
        value = selected.get(field)
        if type(value) is not str or len(value) != 64 or any(
                letter not in '0123456789abcdef' for letter in value):
            raise ValueError(f'approved {field} is missing or invalid')
    return selected


def generate_manifest(*, sample_root, baseline_root, evidence_root,
                      result_root, output, production_commit):
    """Read frozen inputs and write the 183 config files plus their allowlist."""
    if (len(production_commit) != 40 or any(
            character not in '0123456789abcdef' for character in production_commit)):
        raise ValueError('production commit must be a full Git SHA')
    current = subprocess.check_output(
        ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
        text=True).strip()
    if current != production_commit:
        raise ValueError('manifest generator checkout is not pinned production code')
    sample_root, baseline_root = Path(sample_root), Path(baseline_root)
    evidence_root = Path(evidence_root)
    entries = []
    source_cache, data_cache, holdout_cache, validation_cache = {}, {}, {}, {}
    for job_key in sorted(expected_job_keys()):
        dataset, method, alpha_text, seed_text = job_key.split('/')
        alpha = None if alpha_text == 'none' else float(alpha_text)
        seed = int(seed_text)
        source_path = source_config_path(sample_root, baseline_root,
                                         dataset, method)
        if source_path not in source_cache:
            source_cache[source_path] = (
                json.loads(source_path.read_text(encoding='utf-8')),
                _sha256(source_path),
            )
        source, source_sha = source_cache[source_path]
        holdout_path = (evidence_root / 'holdout' /
                        f'{"upmc" if dataset == "upmc_food101" else dataset}'
                        '_tuning_holdout_seed20261023.json')
        if dataset not in holdout_cache:
            locked = json.loads(holdout_path.read_text(encoding='utf-8'))
            if (locked.get('sha256') != HOLDOUT_SELECTED_SHA256[dataset] or
                    locked.get('source_sha256') != DATA_SHA256[dataset]):
                raise ValueError('tuning holdout differs from exact-source anchor')
            holdout_cache[dataset] = _sha256(holdout_path)
        validation_path = (sample_root / 'full' / dataset / 'adaptive' /
                           '42' / 'run' / 'validation' /
                           'validation_manifest.json')
        if dataset not in validation_cache:
            validation_cache[dataset] = _sha256(validation_path)
            if validation_cache[dataset] != VALIDATION_FILE_SHA256[dataset]:
                raise ValueError('paired validation source differs')
        if method in BASELINES[dataset]:
            own_validation = (source_path.parent / 'validation' /
                              'validation_manifest.json')
            if _sha256(own_validation) != validation_cache[dataset]:
                raise ValueError('baseline validation pool is not paired')
        config = build_job_config(
            source, dataset=dataset, method=method, alpha=alpha, seed=seed,
            root=result_root, source_config=source_path,
            holdout_manifest=holdout_path,
            paired_validation_manifest=validation_path,
        )
        source_data = Path(config['development_source_data_path'])
        if source_data not in data_cache:
            data_cache[source_data] = _sha256(source_data)
        if data_cache[source_data] != DATA_SHA256[dataset]:
            raise ValueError('development data differs from frozen source')
        alpha_file = 'none' if alpha is None else str(alpha).replace('.', 'p')
        row = {
            'job_key': job_key, 'dataset': dataset, 'method': method,
            'alpha': alpha, 'seed': seed,
            'config_relpath': f'configs/{dataset}_{method}_{alpha_file}_{seed}.json',
            'config_sha256': hashlib.sha256(_canonical_bytes(config)).hexdigest(),
            'source_config_sha256': source_sha,
            'source_data_sha256': data_cache[source_data],
            'holdout_sha256': holdout_cache[dataset],
            'validation_manifest_sha256': validation_cache[dataset],
            'production_commit': production_commit,
        }
        entries.append((row, config))
    if len(entries) != 183 or {row['job_key'] for row, _ in entries} != expected_job_keys():
        raise AssertionError('generated development matrix differs')
    return write_manifest_files(output, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production-commit', required=True)
    args = parser.parse_args()
    print(generate_manifest(
        sample_root=args.sample_root, baseline_root=args.baseline_root,
        evidence_root=args.evidence_root, result_root=args.result_root,
        output=args.output, production_commit=args.production_commit,
    ))


if __name__ == '__main__':
    main()
