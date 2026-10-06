"""Main entrypoint with multi-seed support and mean±std aggregation."""
import os, sys, json, copy
import numpy as np
from datetime import datetime
from config import get_config
from runner import run_experiment, run_oracle
from utils_logging import tee_to_file

BASELINES = [
    # Baseline lower/middle bound
    ('finetune',     'retrain',         'FineTune (lower bound, CL only)'),
    ('er',           'retrain',         'ER (middle reference, CL only)'),
    # Prototype-based CL
    ('proto_aug',    'retrain',         'PASS+VFL (CL only)'),
    ('proto_evolve', 'retrain',         'V-LETO (CL only)'),
    ('proto_fedspace','retrain',        'FedSpace+VFL (CL only)'),
    # Replay-based CL
    ('der_pp',       'retrain',         'DER++ (CL only)'),
    ('er_ace',       'retrain',         'ER-ACE (CL only)'),
    # CL × UL combinations (full benchmark)
    ('proto_evolve', 'luv',             'V-LETO x LUV'),
    ('proto_evolve', 'gradient_ascent', 'V-LETO x GA'),
    ('proto_aug',    'luv',             'PASS+VFL x LUV'),
    ('proto_fedspace','luv',            'FedSpace+VFL x LUV'),
    ('er',           'luv',             'ER x LUV'),
    ('finetune',     'luv',             'FineTune x LUV'),
    ('finetune',     'gradient_ascent', 'FineTune x GA'),
    ('der_pp',       'luv',             'DER++ x LUV'),
    ('er_ace',       'luv',             'ER-ACE x LUV'),
    # Federated unlearning baselines (Batch 1: prototype/feature-level + projection)
    ('proto_evolve', 'mode',            'V-LETO x MoDe (IoT 2024)'),
    ('proto_evolve', 'fucrt',           'V-LETO x FUCRT (ICCV 2025)'),
    ('proto_evolve', 'fedup',           'V-LETO x FedUP (TSC 2025)'),
    ('proto_evolve', 'fedosd',          'V-LETO x FedOSD (AAAI 2025)'),
    # Federated unlearning baselines (Batch 2: linear-op + conv pruning)
    ('proto_evolve', 'fedau',           'V-LETO x FedAU (Pan 2022)'),
    ('proto_evolve', 'fudp',            'V-LETO x FUDP (WWW 2022)'),
]


def parse_seeds(args):
    """Return list of seeds. If --seeds given, use it; else single-element [args.seed]."""
    if args.seeds.strip():
        return [int(s) for s in args.seeds.split(',') if s.strip()]
    return [args.seed]


def aggregate_seed_results(per_seed_results):
    """Aggregate per-seed result dicts into mean±std summary.

    Each per_seed result is the dict returned by run_experiment/run_oracle.
    """
    if not per_seed_results:
        return {}
    n_seeds = len(per_seed_results)
    # Collect numeric metrics across seeds
    cl_metrics_seeds = [r.get('cl_metrics', {}) for r in per_seed_results]
    aggregated_cl = {}
    for key in ['AA_cil', 'AA_ul', 'AA_final', 'BWT', 'AA']:
        vals = [m.get(key) for m in cl_metrics_seeds if m.get(key) is not None]
        if vals:
            aggregated_cl[key] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'values': [round(float(v), 4) for v in vals],
            }

    # UL metrics — use the LAST UL event per seed (consistent with prior code)
    last_ul = []
    for r in per_seed_results:
        ulm = r.get('ul_metrics', [])
        if ulm:
            last_ul.append(ulm[-1])
    aggregated_ul = {}
    if last_ul:
        for key in ['forget_acc', 'retain_acc', 'mia_score']:
            vals = [u.get(key) for u in last_ul if u.get(key) is not None]
            if vals:
                aggregated_ul[key] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'std': round(float(np.std(vals)), 4),
                    'values': [round(float(v), 4) for v in vals],
                }

    # END-OF-STREAM audit (the benchmark's UA/MIA columns): final_ul_eval is
    # measured after ALL events, so it includes relapse from post-UL learning.
    finals = [r.get('final_ul_eval') for r in per_seed_results if r.get('final_ul_eval')]
    aggregated_final = {}
    if finals:
        for key in ['forget_acc', 'retain_acc', 'mia_score']:
            vals = [f.get(key) for f in finals if f.get(key) is not None]
            if vals:
                aggregated_final[key] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'std': round(float(np.std(vals)), 4),
                    'values': [round(float(v), 4) for v in vals],
                }
        # per-forgotten-class UA at stream end + relearn AUC, averaged over seeds
        for key in ['forget_acc_per_class_final', 'relearn_auc_final']:
            per_cls = {}
            for f in finals:
                for c, v in (f.get(key) or {}).items():
                    per_cls.setdefault(str(c), []).append(float(v))
            if per_cls:
                aggregated_final[key] = {c: {'mean': round(float(np.mean(vs)), 4),
                                             'std': round(float(np.std(vs)), 4)}
                                         for c, vs in per_cls.items()}

    # Timing and comm
    total_times = []
    total_comms = []
    for r in per_seed_results:
        total_times.append(sum(t.get('time_seconds', 0) for t in r.get('timing', [])))
        total_comms.append(sum(c.get('megabytes_transmitted', 0) for c in r.get('comm_stats', [])))
    timing_summary = {
        'mean': round(float(np.mean(total_times)), 1),
        'std': round(float(np.std(total_times)), 1),
    } if total_times else {}
    comm_summary = {
        'mean': round(float(np.mean(total_comms)), 1),
        'std': round(float(np.std(total_comms)), 1),
    } if total_comms else {}

    return {
        'n_seeds': n_seeds,
        'seeds': [r.get('config', {}).get('seed', '?') for r in per_seed_results],
        'cl_metrics': aggregated_cl,
        'ul_metrics': aggregated_ul,
        'final_ul_metrics': aggregated_final,
        'total_time_seconds': timing_summary,
        'total_comm_MB': comm_summary,
    }


def fmt_meanstd(d, key, fmt='.3f'):
    """Format 'mean ± std' for a key from aggregated dict, or '-' if missing."""
    if not d or key not in d:
        return '-'
    m, s = d[key].get('mean'), d[key].get('std')
    if m is None:
        return '-'
    if s is None or s == 0:
        return f"{m:{fmt}}"
    return f"{m:{fmt}}±{s:{fmt}}"


def run_one_baseline_multiseed(base_args, cl, ul, desc, parent_dir):
    """Run one (CL, UL) combo across all seeds. Returns aggregated result dict."""
    seeds = parse_seeds(base_args)
    base_args.cl_method, base_args.ul_method = cl, ul
    if not getattr(base_args, 'exp_name', ''):
        base_args.exp_name = f"{cl}_x_{ul}_{base_args.data}"
    combo_dir = os.path.join(parent_dir, base_args.exp_name)
    os.makedirs(combo_dir, exist_ok=True)

    per_seed_results = []
    for s in seeds:
        seed_args = copy.deepcopy(base_args)
        seed_args.seed = s
        seed_args.output_dir = os.path.join(combo_dir, f'seed_{s}')
        if os.path.exists(seed_args.output_dir):
            raise FileExistsError(f"Refusing to reuse existing seed output_dir: {seed_args.output_dir}")
        os.makedirs(seed_args.output_dir, exist_ok=True)
        with open(os.path.join(seed_args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(seed_args), f, indent=2, default=str)
        print(f"\n{'#'*70}\n  {desc}  |  seed={s}\n{'#'*70}")
        log_path = os.path.join(seed_args.output_dir, 'run.log')
        try:
            with tee_to_file(log_path):
                r = run_experiment(seed_args)
            per_seed_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            per_seed_results.append({'error': str(e), 'config': {'seed': s, 'cl_method': cl, 'ul_method': ul}})

    agg = aggregate_seed_results([r for r in per_seed_results if 'error' not in r])
    agg['desc'] = desc
    agg['cl_method'] = cl
    agg['ul_method'] = ul
    agg['n_failed'] = sum(1 for r in per_seed_results if 'error' in r)
    if agg['n_failed'] > 0:
        agg['errors'] = [r['error'] for r in per_seed_results if 'error' in r]
    with open(os.path.join(combo_dir, 'aggregated.json'), 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    return agg


def run_oracle_multiseed(base_args, parent_dir):
    seeds = parse_seeds(base_args)
    base_args.exp_name = f"oracle_{base_args.data}"
    oracle_dir = os.path.join(parent_dir, base_args.exp_name)
    os.makedirs(oracle_dir, exist_ok=True)
    per_seed_results = []
    for s in seeds:
        seed_args = copy.deepcopy(base_args)
        seed_args.seed = s
        seed_args.output_dir = os.path.join(oracle_dir, f'seed_{s}')
        os.makedirs(seed_args.output_dir, exist_ok=True)
        print(f"\n{'#'*70}\n  Oracle  |  seed={s}\n{'#'*70}")
        log_path = os.path.join(seed_args.output_dir, 'run.log')
        try:
            with tee_to_file(log_path):
                r = run_oracle(seed_args)
            per_seed_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            per_seed_results.append({'error': str(e), 'config': {'seed': s}})
    agg = aggregate_seed_results([r for r in per_seed_results if 'error' not in r])
    agg['desc'] = 'Oracle'
    agg['cl_method'] = 'oracle'
    agg['ul_method'] = 'oracle'
    with open(os.path.join(oracle_dir, 'aggregated.json'), 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    return agg


def inject_kl_to_oracle(all_aggs, sdir, base_data, seeds):
    """Post-hoc KL: compare each method's saved final-state output distribution
    against the Oracle's (same seed), on the shared retained test set."""
    from metrics import compute_kl_to_oracle
    oracle_dir = os.path.join(sdir, f'oracle_{base_data}')
    for a in all_aggs:
        a.setdefault('cl_metrics', {})
        if a.get('cl_method') == 'oracle':
            a['cl_metrics']['KL'] = {'mean': 0.0, 'std': 0.0, 'values': [0.0] * len(seeds)}
            continue
        combo = f"{a.get('cl_method')}_x_{a.get('ul_method')}_{base_data}"
        vals = []
        for s in seeds:
            op = os.path.join(oracle_dir, f'seed_{s}', 'final_probs.npz')
            mp = os.path.join(sdir, combo, f'seed_{s}', 'final_probs.npz')
            if os.path.exists(op) and os.path.exists(mp):
                od, md = np.load(op), np.load(mp)
                kl = compute_kl_to_oracle(od['probs'], md['probs'], od['retained'].tolist())
                if kl is not None:
                    vals.append(kl)
        if vals:
            a['cl_metrics']['KL'] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'values': [round(v, 4) for v in vals],
            }


def print_summary_table(all_aggs):
    print(f"\n{'='*132}")
    print(f"{'Method':<32} {'AA_cil':>16} {'AA_final':>16} {'BWT':>14} {'KL':>12} {'F-Acc':>14} {'MIA':>14} {'Time(s)':>10}")
    print('-' * 132)
    for a in all_aggs:
        desc = a.get('desc', '?')[:31]
        cl = a.get('cl_metrics', {})
        ul = a.get('ul_metrics', {})
        t = a.get('total_time_seconds', {})
        time_str = fmt_meanstd({'_': t}, '_', '.0f') if t else '-'
        print(f"{desc:<32} "
              f"{fmt_meanstd(cl,'AA_cil'):>16} "
              f"{fmt_meanstd(cl,'AA_final'):>16} "
              f"{fmt_meanstd(cl,'BWT'):>14} "
              f"{fmt_meanstd(cl,'KL','.4f'):>12} "
              f"{fmt_meanstd(ul,'forget_acc'):>14} "
              f"{fmt_meanstd(ul,'mia_score'):>14} "
              f"{time_str:>10}")
    print('=' * 132)


def run_all():
    base = get_config()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sdir = os.path.join(base.results_dir, f'full_benchmark_{ts}')
    os.makedirs(sdir, exist_ok=True)
    master_log = os.path.join(sdir, 'master.log')
    print(f"Multi-seed benchmark, seeds = {parse_seeds(base)}, output → {sdir}")
    print(f"Master log → {master_log}")

    with tee_to_file(master_log):
        print(f"Benchmark started at {ts}")
        print(f"Seeds: {parse_seeds(base)}")
        print(f"Data: {base.data}, replay_mode: {base.replay_mode}")

        all_aggs = []
        for i, (cl, ul, desc) in enumerate(BASELINES):
            print(f"\n\n{'$'*80}\n  Baseline {i+1}/{len(BASELINES)+1}: {desc}\n{'$'*80}")
            agg = run_one_baseline_multiseed(base, cl, ul, desc, sdir)
            all_aggs.append(agg)

        # Oracle
        print(f"\n\n{'$'*80}\n  Baseline {len(BASELINES)+1}/{len(BASELINES)+1}: Oracle\n{'$'*80}")
        agg = run_oracle_multiseed(base, sdir)
        all_aggs.append(agg)

        # KL-to-Oracle (post-hoc; Oracle reference now exists for all seeds)
        inject_kl_to_oracle(all_aggs, sdir, base.data, parse_seeds(base))

        summary = {
            'timestamp': ts,
            'seeds': parse_seeds(base),
            'data': base.data,
            'replay_mode': base.replay_mode,
            'num_parties': base.num_parties,
            'aggregation': base.aggregation,
            'baselines': all_aggs,
        }
        with open(os.path.join(sdir, 'benchmark_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print_summary_table(all_aggs)
        print(f"\nSaved to: {sdir}")


if __name__ == '__main__':
    if '--run_all' in sys.argv:
        sys.argv.remove('--run_all')
        run_all()
    else:
        args = get_config()
        # Multi-seed for single combo
        seeds = parse_seeds(args)
        if len(seeds) > 1:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            parent_name = args.exp_name if getattr(args, 'exp_name', '') else f"{args.cl_method}_x_{args.ul_method}_{args.data}"
            parent = os.path.join(args.results_dir, f"{parent_name}_multiseed_{ts}")
            os.makedirs(parent, exist_ok=True)
            agg = run_one_baseline_multiseed(args, args.cl_method, args.ul_method,
                                              f"{args.cl_method} x {args.ul_method}", parent)
            print_summary_table([agg])
        else:
            run_experiment(args)
