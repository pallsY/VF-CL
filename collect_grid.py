"""Collect the VFL-CLU benchmark grid into per-(dataset,order) summary tables.

Walks results/grid_tab/<ds_o>/<cl>__<ul>/, dedupes duplicate multiseed dirs
(double-dispatch leftovers) by preferring the LATEST dir with >=3 complete
seeds, aggregates per-seed results.json, and emits:
  - stdout matrix digests (RA / UA / MIA / #P) for quick inspection
  - results/grid_tab/summary_<ds_o>.json with every cell's metrics
Metrics follow docs_benchmark_protocol.md: RA/UA/MIA are END-OF-STREAM
(final_ul_eval, relapse-inclusive); #P and RTE from per-event records.
"""
import json, os, sys, glob, collections
import numpy as np

CL_ORDER = ['finetune', 'ewc', 'lwf', 'lwf_wa', 'lwf_fim', 'afc', 'gpm',
            'adagauss', 'target', 'proto_evolve', 'proto_fedspace', 'ours_protoOC']
UL_ORDER = ['retrain', 'ga', 'luv', 'mode', 'fucrt', 'fedup', 'fedosd', 'fedau',
            'ours_localrt']


def best_multiseed_dir(combo_dir):
    """Latest multiseed run having the most complete seed set."""
    cands = []
    for ms in sorted(glob.glob(os.path.join(combo_dir, '*_multiseed_*'))):
        seeds = glob.glob(os.path.join(ms, '*', 'seed_*', 'results.json'))
        if seeds:
            cands.append((len(seeds), ms, seeds))
    if not cands:
        return None, []
    n = max(c[0] for c in cands)
    full = [c for c in cands if c[0] == n]
    return full[-1][1], full[-1][2]        # latest among the most-complete


def collect_combo(combo_dir):
    _, seed_files = best_multiseed_dir(combo_dir)
    if not seed_files:
        return None
    per = collections.defaultdict(list)
    for sf in sorted(seed_files):
        d = json.load(open(sf))
        cm = d.get('cl_metrics') or {}
        fe = d.get('final_ul_eval') or {}
        ulm = d.get('ul_metrics') or []
        per['AA'].append(cm.get('AA_final', cm.get('AA')))
        per['BWT'].append(cm.get('BWT'))
        per['RA'].append(fe.get('retain_acc'))
        per['UA'].append(fe.get('forget_acc'))
        auc = fe.get('relearn_auc_final') or {}
        per['MIA'].append(np.mean(list(auc.values())) if auc else None)
        pt = [u.get('parties_touched') for u in ulm if u.get('parties_touched') is not None]
        per['P'].append(np.mean(pt) if pt else None)
        per['RTE'].append(sum(t.get('time_seconds', 0) for t in d.get('timing', [])))
        # relapse: max over forgotten classes of (UA at end - UA right after its event)
        ev_ua = {}
        for u in ulm:
            for c, _ in (fe.get('forget_acc_per_class_final') or {}).items():
                pass
        per_cls_end = fe.get('forget_acc_per_class_final') or {}
        # per-event forget_acc covers cumulative set; use per-class end vs 0 proxy
        per['UA_max_cls'].append(max(per_cls_end.values()) if per_cls_end else None)
    out = {}
    for k, vs in per.items():
        vs = [v for v in vs if v is not None]
        if vs:
            out[k] = {'mean': round(float(np.mean(vs)), 4),
                      'std': round(float(np.std(vs)), 4), 'n': len(vs)}
    return out


def digest(ds_dir):
    name = os.path.basename(ds_dir.rstrip('/'))
    cells = {}
    for combo_dir in sorted(glob.glob(os.path.join(ds_dir, '*__*'))):
        combo = os.path.basename(combo_dir)
        cl, ul = combo.split('__', 1)
        m = collect_combo(combo_dir)
        if m:
            cells[(cl, ul)] = m
    if not cells:
        return
    with open(os.path.join(os.path.dirname(ds_dir), f'summary_{name}.json'), 'w') as f:
        json.dump({f'{cl}__{ul}': m for (cl, ul), m in cells.items()}, f, indent=1)

    for metric, fmt in [('RA', '{:.2f}'), ('UA', '{:.2f}'), ('MIA', '{:.2f}'), ('P', '{:.1f}')]:
        print(f'\n== {name} : {metric} ==')
        hdr = f'{"":<14}' + ''.join(f'{u:>9}' for u in UL_ORDER)
        print(hdr)
        for cl in CL_ORDER:
            row = f'{cl:<14}'
            for ul in UL_ORDER:
                m = cells.get((cl, ul), {}).get(metric)
                row += f'{fmt.format(m["mean"]) if m else "-":>9}'
            print(row)
    n_missing = sum(1 for cl in CL_ORDER for ul in UL_ORDER if (cl, ul) not in cells)
    print(f'\n[{name}] combos: {len(cells)}  missing: {n_missing}')


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'results/grid_tab'
    for ds_dir in sorted(glob.glob(os.path.join(root, '*_o[12]'))):
        digest(ds_dir)
