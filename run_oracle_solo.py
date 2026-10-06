"""Run only Oracle (Joint-RT) as a multi-seed combo. Same CLI as main.py.

The default main.py runs Oracle inside --run_all; for the fragmentation sweep
we need to evaluate Oracle independently at each fragmentation level (different
--num_parties), and don't want to re-run the full 22-baseline matrix each time.
"""
import os, json, copy
import numpy as np
from datetime import datetime
from config import get_config
from runner import run_oracle
from main import aggregate_seed_results, parse_seeds
from utils_logging import tee_to_file


def main():
    args = get_config()
    seeds = parse_seeds(args)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"oracle_{args.data}"
    parent = os.path.join(args.results_dir, f"{args.exp_name}_multiseed_{ts}")
    os.makedirs(parent, exist_ok=True)

    per_seed = []
    for s in seeds:
        sa = copy.deepcopy(args)
        sa.seed = s
        sa.output_dir = os.path.join(parent, f'seed_{s}')
        os.makedirs(sa.output_dir, exist_ok=True)
        print(f"\n{'#'*70}\n  Oracle | seed={s}\n{'#'*70}")
        log_path = os.path.join(sa.output_dir, 'run.log')
        try:
            with tee_to_file(log_path):
                r = run_oracle(sa)
            per_seed.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            per_seed.append({'error': str(e), 'config': {'seed': s}})

    agg = aggregate_seed_results([r for r in per_seed if 'error' not in r])
    agg['desc'] = 'Oracle'
    agg['cl_method'] = 'oracle'
    agg['ul_method'] = 'oracle'
    agg['n_failed'] = sum(1 for r in per_seed if 'error' in r)
    if agg['n_failed'] > 0:
        agg['errors'] = [r['error'] for r in per_seed if 'error' in r]
    with open(os.path.join(parent, 'aggregated.json'), 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\nOracle done -> {parent}")


if __name__ == '__main__':
    main()
