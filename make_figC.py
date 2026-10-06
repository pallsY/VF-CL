"""Figure C — certificate-driven scrub (the certificate->action loop).

The head-reconnection certificate revealed that the cosine-CE-only scrub
optimizes a PROXY (normalized deployed logit) and leaves the RAW per-party score
<W_f^k,e_k> — the certified quantity — intact, so eps stays large and the bound
is vacuous. Adding a direct raw-score-suppression term (--roar_scrub_raw_weight)
makes the scrub minimize the certified evidence itself. This plots eps, E|R_f|,
the Thm-1 bound, and retain accuracy vs the raw-suppression weight on Covertype
class 6 (3 seeds): leakage collapses (eps 6.0 -> ~1.1) at ~2pt retain cost.

Usage: python make_figC.py --root results/covtype_raw --fclass 6 --out figs/covtype
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/covtype_raw')
    ap.add_argument('--fclass', default='6')
    ap.add_argument('--out', default='figs/covtype')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fc = args.fclass

    agg = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f'{args.root}/rw_*/**/roar_ownership_f*.json', recursive=True):
        rw = float(p.split('/rw_')[1].split('/')[0].replace('p', '.'))
        rd = json.load(open(p)).get('residual_diag', {})
        if fc in rd:
            agg[rw]['eps'].append(rd[fc]['eps'])
            agg[rw]['R'].append(rd[fc]['R_f_measured_abs'])
            agg[rw]['B'].append(rd[fc]['bound_rhs'])
    for p in glob.glob(f'{args.root}/rw_*/**/results.json', recursive=True):
        rw = float(p.split('/rw_')[1].split('/')[0].replace('p', '.'))
        agg[rw]['ret'].append(json.load(open(p))['ul_metrics'][-1]['retain_acc'])
    if not agg:
        print(f'no data under {args.root}'); return

    rws = sorted(agg); xs = [str(r) for r in rws]
    mean = lambda r, k: np.mean(agg[r][k]) if agg[r][k] else np.nan
    std = lambda r, k: np.std(agg[r][k]) if agg[r][k] else 0
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable'); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(xs, [mean(r, 'eps') for r in rws], yerr=[std(r, 'eps') for r in rws],
                 marker='o', lw=2, capsize=3, color='C3', label=r'$\epsilon$ (post-scrub tolerance)')
    ax1.errorbar(xs, [mean(r, 'R') for r in rws], yerr=[std(r, 'R') for r in rws],
                 marker='s', lw=2, capsize=3, color='C0', label=r'$\mathbb{E}|R_f|$ (residual leakage)')
    ax1.plot(xs, [mean(r, 'B') for r in rws], '--', color='C0', alpha=.6, label='Thm-1 bound')
    ax1.set_xlabel('scrub raw-suppression weight'); ax1.set_ylabel(f'class-{fc} residual evidence')
    ax1.set_title('(c) Certificate-driven scrub: leakage collapses'); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
    ax2.errorbar(xs, [mean(r, 'ret') for r in rws], yerr=[std(r, 'ret') for r in rws],
                 marker='D', lw=2, capsize=3, color='C2')
    ax2.set_xlabel('scrub raw-suppression weight'); ax2.set_ylabel('retain accuracy')
    ax2.set_title('(c2) Utility cost is small'); ax2.grid(alpha=.3)
    fig.tight_layout()
    png = os.path.join(args.out, 'figC_certificate_action.png')
    fig.savefig(png, dpi=160, bbox_inches='tight')
    print(f'wrote {png}')
    for r in rws:
        print(f'raw={r:>5} eps={mean(r,"eps"):.2f} E|R|={mean(r,"R"):.2f} retain={mean(r,"ret"):.3f}')


if __name__ == '__main__':
    main()
