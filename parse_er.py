#!/usr/bin/env python3
"""Parse ER sweep results into per-task trajectory matrices.

Usage: python parse_er.py results/after_c300 [results/before_c20 ...]
Each arg is a results dir; we glob */results.json under it.
"""
import json, sys, glob, os


def load(run_dir):
    hits = glob.glob(os.path.join(run_dir, '**', 'results.json'), recursive=True)
    if not hits:
        return None
    # newest by mtime
    f = max(hits, key=os.path.getmtime)
    return f, json.load(open(f))


def fmt_run(name, run_dir):
    got = load(run_dir)
    if got is None:
        print(f"\n### {name} ({run_dir}): NO results.json yet")
        return
    f, d = got
    hist = d['task_acc_history']
    n = len(hist)
    cm = d.get('cl_metrics', {})
    print(f"\n### {name}  [{run_dir}]")
    # full args are in the sibling config.json (results.json['config'] is a summary)
    cfg = {}
    cj = os.path.join(os.path.dirname(f), 'config.json')
    if os.path.exists(cj):
        cfg = json.load(open(cj))
    print(f"    er_per_class={cfg.get('er_per_class')}  er_batch={cfg.get('er_batch')}  "
          f"er_alpha={cfg.get('er_alpha')}  epochs={cfg.get('epochs_per_task')}")
    # Triangular trajectory line (rows = state after task i)
    parts = []
    for i, ev in enumerate(hist):
        accs = ev['per_task_accs']
        cols = ' '.join(f"{accs.get(f'task_{j}', float('nan')):.2f}" for j in range(i + 1))
        parts.append(f"t{i}:{cols}")
    print("    traj: " + ' | '.join(parts))
    # Aligned grid: rows = after task i, cols = task_0..task_{n-1}
    header = "          " + ''.join(f"  tsk{j}" for j in range(n))
    print(header)
    for i, ev in enumerate(hist):
        accs = ev['per_task_accs']
        row = ''.join(
            (f"  {accs[f'task_{j}']:.2f}" if f'task_{j}' in accs else "   -- ")
            for j in range(n))
        print(f"    after t{i}:{row}   (overall {ev['overall_acc']:.4f})")
    # Final-row retention = mean of old-task cols at last event
    last = hist[-1]['per_task_accs']
    olds = [last[f'task_{j}'] for j in range(n - 1) if f'task_{j}' in last]
    ret = sum(olds) / len(olds) if olds else float('nan')
    print(f"    AA_final={cm.get('AA_final')}  AA_cil={cm.get('AA_cil')}  "
          f"BWT={cm.get('BWT')}  mean_old_task_acc_final={ret:.4f}")


if __name__ == '__main__':
    for run_dir in sys.argv[1:]:
        name = os.path.basename(run_dir.rstrip('/'))
        fmt_run(name, run_dir)
