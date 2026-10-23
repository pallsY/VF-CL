"""Read-only source check for paired development datasets; never trains."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from party_drift_manifest import load_approved_job
from run_party_drift_development import validate_job_sources, _sha256
from run_resource_lite_pilot import PilotDataset


CHECK_KEYS = (
    'upmc_food101/drift/0.25/70',
    'upmc_food101/derpp/none/70',
    'upmc_food101/icarl/none/70',
    'isolet/drift/0.25/70',
    'isolet/icarl/none/70',
    'cifar100/drift/0.25/70',
    'cifar100/er_ace/none/70',
)


def check(manifest, digest, output):
    manifest, output = Path(manifest).resolve(), Path(output)
    checkout = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    paired = {}
    for key in CHECK_KEYS:
        row = load_approved_job(key, manifest, digest,
                                production_commit=commit)
        config = json.loads((manifest.parent / row['config_relpath']).read_text(
            encoding='utf-8'))
        validate_job_sources(config, row)
        folder = output / key.replace('/', '_').replace('.', 'p')
        folder.mkdir()
        args = SimpleNamespace(**{
            **config, 'output_dir': str(folder),
            'results_dir': str(output), 'exp_name': folder.name,
        })
        dataset = PilotDataset(args)
        locked = json.loads(Path(
            args.development_holdout_manifest).read_text(encoding='utf-8'))
        if (dataset.holdout_manifest['sha256'] != locked['sha256'] or
                dataset.holdout_indices != set(locked['ordered_indices']) or
                dataset.holdout_indices & dataset.pilot_pair_excluded):
            raise ValueError(f'paired holdout differs for {key}')
        dataset_name = config['pilot_dataset']
        reserved = (tuple(sorted(dataset.pilot_pair_excluded)),
                    tuple(sorted(dataset.holdout_indices)))
        if dataset_name in paired and paired[dataset_name] != reserved:
            raise ValueError(f'paired train/holdout indices differ in {dataset_name}')
        paired[dataset_name] = reserved
        rows.append({
            'job_key': key, 'holdout_selected_sha256': locked['sha256'],
            'holdout_count': len(dataset.holdout_indices),
            'paired_reserved_count': len(dataset.pilot_pair_excluded),
            'generated_manifest_sha256': _sha256(
                folder / 'pilot_holdout' / 'manifest.json'),
            'test_loader_created': False,
        })
    gate = {'status': 'paired_dataset_pass', 'manifest_sha256': digest,
            'production_commit': commit, 'checks': rows,
            'test_loader_created': False}
    with (output / 'DATASET_GATE.json').open('x', encoding='utf-8') as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': gate['status'], 'checks': len(rows)},
                     sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.manifest, args.manifest_sha256, args.output)
