import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

from config import validate_party_kd_variant


KEY_FIELDS = [
    'dep_tracking_enabled',
    'party_kd_enabled',
    'party_kd_mode',
    'party_kd_lambda',
    'party_proto_enabled',
    'party_proto_mode',
    'cl_method',
    'ul_method',
    'data',
    'num_tasks',
    'classes_per_task',
    'num_parties',
    'model_type',
    'aggregation',
    'epochs_per_task',
    'batch_size',
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_head_tail(path, n=5):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()
    head = lines[:n]
    tail = lines[-n:] if len(lines) >= n else lines
    return head, tail


def normalize_cfg(cfg):
    return {k: cfg.get(k) for k in KEY_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--run_glob', default='',
                    help='Optional config.json glob relative to results_root, e.g. */config.json')
    ap.add_argument('--expected_party_kd_variant', default='',
                    choices=['', 'base', 'uniform', 'static', 'inverse', 'shuffled'])
    args = ap.parse_args()

    groups = defaultdict(list)
    failures = []
    audit_failed = False
    if args.run_glob:
        config_paths = glob.glob(os.path.join(args.results_root, args.run_glob), recursive=True)
        seed_dirs = sorted({os.path.dirname(path) for path in config_paths})
    else:
        seed_dirs = sorted(glob.glob(os.path.join(
            args.results_root, '*multiseed*', 'proto_evolve_x_retrain_cifar100', 'seed_*')))

    for seed_dir in seed_dirs:
        cfg_path = os.path.join(seed_dir, 'config.json')
        res_path = os.path.join(seed_dir, 'results.json')
        log_path = os.path.join(seed_dir, 'run.log')
        require_log = not args.run_glob
        if not (os.path.exists(cfg_path) and os.path.exists(res_path)
                and (os.path.exists(log_path) or not require_log)):
            failures.append({
                'seed_dir': seed_dir,
                'missing': {
                    'config.json': not os.path.exists(cfg_path),
                    'results.json': not os.path.exists(res_path),
                    'run.log': require_log and not os.path.exists(log_path),
                }
            })
            continue
        cfg = json.load(open(cfg_path))
        try:
            validate_party_kd_variant(cfg, args.expected_party_kd_variant)
        except ValueError as exc:
            failures.append({'seed_dir': seed_dir, 'variant_error': str(exc)})
            continue
        res = json.load(open(res_path))
        key = tuple((k, cfg.get(k)) for k in KEY_FIELDS if k != 'seed')
        head, tail = read_head_tail(log_path) if os.path.exists(log_path) else ([], [])
        groups[key].append({
            'seed_dir': seed_dir,
            'seed': cfg.get('seed'),
            'cfg': normalize_cfg(cfg),
            'AA_final': res.get('cl_metrics', {}).get('AA_final'),
            'AA_cil': res.get('cl_metrics', {}).get('AA_cil'),
            'BWT': res.get('cl_metrics', {}).get('BWT'),
            'AA_final_taskil': res.get('cl_metrics', {}).get('AA_final_taskil'),
            'AA_cil_taskil': res.get('cl_metrics', {}).get('AA_cil_taskil'),
            'log_size': os.path.getsize(log_path) if os.path.exists(log_path) else 0,
            'log_sha256': sha256(log_path) if os.path.exists(log_path) else '',
            'log_head': head,
            'log_tail': tail,
        })

    print('AUDIT_STAGE2_REPORT')
    if failures:
        audit_failed = True
        print('FAILURES')
        for x in failures:
            print('  seed_dir', x['seed_dir'])
            if 'variant_error' in x:
                print('  STATUS: FAIL_VARIANT_CONTRACT')
                print('  variant_error', x['variant_error'])
            else:
                print('  missing', x['missing'])
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        items = sorted(items, key=lambda x: x['seed'])
        print('GROUP_KEY', key)
        cfgs = [json.dumps(x['cfg'], sort_keys=True) for x in items]
        cfg_consistent = len(set(cfgs)) == 1
        print('  cfg_consistent_except_seed', cfg_consistent)
        if not cfg_consistent:
            audit_failed = True
            print('  STATUS: FAIL_CONFIG_MISMATCH')
        else:
            print('  STATUS: CONFIG_OK')
        for x in items:
            print(f"  seed={x['seed']} dir={x['seed_dir']}")
            print(f"    AA_final={x['AA_final']:.4f} AA_cil={x['AA_cil']:.4f} BWT={x['BWT']:.4f} AA_final_taskil={x['AA_final_taskil']:.4f} AA_cil_taskil={x['AA_cil_taskil']:.4f}")
            print(f"    runlog_size={x['log_size']} sha256={x['log_sha256']}")
            print(f"    head={x['log_head']}")
            print(f"    tail={x['log_tail']}")
        # Conservative warning: identical log hashes across seeds are suspicious
        log_hashes = [x['log_sha256'] for x in items if x['log_sha256']]
        if len(set(log_hashes)) != len(log_hashes):
            print('  WARNING: duplicate run.log hashes across seeds -> requires manual review')
        # Conservative warning: if top-level parent directory timestamp is shared across groups,
        # keep it visible for manual review rather than silently blessing the data.
        parents = {os.path.dirname(os.path.dirname(x['seed_dir'])) for x in items}
        print('  parent_dirs', sorted(parents))

    if audit_failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
