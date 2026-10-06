"""Figure (CL leg) — unlearning-aware continual learning.

The ownership-concentration regularizer (--own_concentrate_weight) shapes the
continual model so each class is owned by ~one party. This makes ANY future
certified unlearning cheaper: the minimal owning set |S*| shrinks toward 1 and
the certificate's coverage-leakage term (1-tau)Z_f (evidence on un-scrubbed
parties) collapses — at a small continual-accuracy cost, with unlearning quality
held at the retrain-oracle AUC floor. This is the forward CL->UL coupling that
makes the continual and unlearning halves of the framework load-bearing on each
other.

Usage: python make_figCL.py --root results/ocw_cl --fclass 4 --out figs/cl_aware
"""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/ocw_cl')
    ap.add_argument('--fclass', default='4')
    ap.add_argument('--out', default='figs/cl_aware')
    args = ap.parse_args()
    import os; os.makedirs(args.out, exist_ok=True)
    fc = args.fclass

    def ocw_of(p):
        m = re.search(r'/ocw_([0-9p]+)/', p.replace('/ocw_cl/', '/'))
        return float(m.group(1).replace('p', '.')) if m else None

    ag = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f'{args.root}/ocw_*/**/roar_ownership_f*.json', recursive=True):
        o = ocw_of(p); d = json.load(open(p)); rd = d.get('residual_diag', {})
        if fc not in rd:
            continue
        sh = np.array(d['ownership'][fc]['shares'], float); pe = sh / sh.sum()
        ag[o]['ent'].append(float(-(pe*np.log(pe+1e-12)).sum()/np.log(len(pe))))
        ag[o]['S'].append(len(d['ownership'][fc]['S_star']))
        ag[o]['res'].append(rd[fc]['residual_term_outside_S'])
        ag[o]['auc'].append(rd[fc].get('relearn_attack', {}).get('auc', np.nan))
    for p in glob.glob(f'{args.root}/ocw_*/**/results.json', recursive=True):
        o = ocw_of(p); d = json.load(open(p)); ulm = d.get('ul_metrics', [])
        if ulm:
            ag[o]['retain'].append(ulm[-1]['retain_acc'])
    if not ag:
        print('no data'); return
    xs = sorted(ag); m = lambda o, k: float(np.nanmean(ag[o][k])) if ag[o][k] else np.nan
    summary = {str(o): {k: m(o, k) for k in ['ent', 'S', 'res', 'auc', 'retain']} for o in xs}
    json.dump(summary, open(f'{args.out}/figCL_summary.json', 'w'), indent=2)
    print(f'{"ocw":>5} {"entropy":>7} {"|S*|":>5} {"cover-leak":>10} {"retain":>7} {"AUC":>5}')
    for o in xs:
        print(f'{o:>5} {m(o,"ent"):>7.2f} {m(o,"S"):>5.1f} {m(o,"res"):>10.3f} {m(o,"retain"):>7.3f} {m(o,"auc"):>5.2f}')

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; wrote summary only'); return
    xlabels = [str(o) for o in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xlabels, [m(o, 'S') for o in xs], 'o-', color='C0', lw=2, label=r'$|S^*|$ (comm, parties)')
    ax.plot(xlabels, [m(o, 'res') for o in xs], 's-', color='C3', lw=2, label=r'coverage leakage $(1-\tau)Z_f$')
    ax.set_xlabel('ownership-concentration weight (unlearning-aware CL)')
    ax.set_ylabel('future-unlearning cost')
    ax.grid(alpha=.3); ax.legend(loc='upper right', fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(xlabels, [m(o, 'retain') for o in xs], 'D--', color='C2', lw=2, label='retain acc (CL cost)')
    ax2.set_ylabel('retain accuracy', color='C2'); ax2.set_ylim(0.5, 0.75)
    ax2.legend(loc='lower left', fontsize=8)
    ax.set_title('Unlearning-aware CL: concentrate ownership → cheaper future unlearning')
    fig.tight_layout(); fig.savefig(f'{args.out}/figCL_tradeoff.png', dpi=160, bbox_inches='tight')
    print(f'wrote {args.out}/figCL_tradeoff.png')


if __name__ == '__main__':
    main()
