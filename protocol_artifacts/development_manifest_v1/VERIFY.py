import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = root / 'manifest.jsonl'
expected_sha = '532f551f74e8db2e3f6af088d0bbf3995d9426bec822b357dd7726d4d0734c20'
producer = 'b2b9546f82765c25e18394874ad0f79e786ffb82'
result_root = '/brf/users/yangxx/VFCL-results/party-drift-development-v1-20261024'
actual_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert actual_sha == expected_sha == (root / 'manifest.sha256').read_text().strip()
rows = [json.loads(line) for line in manifest.read_text().splitlines()]
assert len(rows) == 183
expected = set()
baselines = {'cifar100': ('er_ace',), 'isolet': ('icarl',),
             'upmc_food101': ('derpp', 'icarl')}
replay = ('all_party', 'uniform', 'static', 'drift', 'cmcdr', 'cmcdr_budget')
for dataset in baselines:
    for seed in (70, 71, 72):
        for method in ('sample_only_off',) + baselines[dataset]:
            expected.add(f'{dataset}/{method}/none/{seed}')
        for method in replay:
            for alpha in (0.25, 0.5, 1.0):
                expected.add(f'{dataset}/{method}/{alpha}/{seed}')
assert {row['job_key'] for row in rows} == expected
assert len(set(row['job_key'] for row in rows)) == 183
seen_configs = set()
by_dataset = {}
for row in rows:
    path = (root / row['config_relpath']).resolve()
    assert path.is_relative_to((root / 'configs').resolve())
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['config_sha256']
    config = json.loads(path.read_text())
    dataset, method, alpha, seed = row['job_key'].split('/')
    assert row['dataset'] == dataset and row['method'] == method
    assert row['seed'] == config['seed'] == int(seed)
    assert row['alpha'] == (None if alpha == 'none' else float(alpha))
    assert row['production_commit'] == producer
    assert config['pilot_protocol'] == 'party-drift-development-v1'
    assert config['pilot_holdout_seed'] == 20261023
    assert config['pilot_holdout_per_class'] == 30
    assert config['formal_deferred_evaluation'] is True
    assert config['output_dir'] == f'{result_root}/{dataset}/{method}/{alpha}/{seed}'
    assert config['device'] == ('cuda:0' if int(seed) % 2 == 0 else 'cuda:1')
    assert config['party_drift_enabled'] == int(method in replay)
    if method in replay:
        assert config['party_replay_arm'] == method
        assert config['party_replay_alpha'] == float(alpha)
    else:
        assert config['party_replay_arm'] == 'off'
    seen_configs.add(path)
    by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
assert len(seen_configs) == 183
assert set((root / 'configs').glob('*.json')) == seen_configs
assert by_dataset == {'cifar100': 60, 'isolet': 60, 'upmc_food101': 63}
print(json.dumps({'status': 'verified', 'rows': 183, 'config_files': 183,
                  'manifest_sha256': actual_sha, 'production_commit': producer,
                  'by_dataset': by_dataset}, sort_keys=True))
