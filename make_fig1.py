"""Figure 1 — Theorem-1 certified residual-leakage validation on mfeat-6view.

Consumes roar's residual_diag (post-scrub head-reconnection probe) across the
roar_tau_own sweep. For each forget class f and tau:

  measured  E|R_f|                 (head-reconnection attack recovers this)
  a-priori bound  |b_f| + eps|S*| + (1-tau)Z_f      (Theorem 1)
  residual term   (1-tau)Z_f = sum_{k notin S*} E|s_{f,k}|   (the tau knob)

Panel (a): measured E|R_f| vs a-priori bound vs residual term, across tau, per f.
           The bound must sit ABOVE the measured curve and be informative
           (not vacuously loose); the residual term shows the tau-controlled
           leakage floor.
Panel (b): certified comm-vs-leakage Pareto — x = communication |S*|/P,
           y = leakage E|R_f| (with deployed forget_acc overlaid). Sweeping tau
           traces the frontier.

Usage:  python make_fig1.py --root ./results/mfeat_tau --out ./figs
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np


def load_runs(root, sweep='tau'):
    """Return list of dicts: one per (sweep_val, seed, forget_class).
    sweep='tau' reads tau_* dirs (mfeat); sweep='rho' reads rho_* dirs (synthvfl,
    where the dir name encodes rho as 0p75 -> 0.75). The swept value is stored under
    key 'tau' regardless so downstream plotting is uniform (x-axis label set by caller).
    """
    rows = []
    for tau_dir in sorted(glob.glob(os.path.join(root, f'{sweep}_*'))):
        raw = os.path.basename(tau_dir).split('_', 1)[1]
        try:
            tau = float(raw.replace('p', '.')) if sweep == 'rho' else float(raw)
        except ValueError:
            continue
        for own_path in glob.glob(os.path.join(tau_dir, '**', 'roar_ownership_f*.json'),
                                  recursive=True):
            with open(own_path) as fh:
                try:
                    d = json.load(fh)
                except json.JSONDecodeError:
                    continue
            rd = d.get('residual_diag')
            if not rd:
                continue
            seed_dir = os.path.dirname(own_path)
            seed = os.path.basename(seed_dir)
            # forget class(es) from filename roar_ownership_f0.json
            for f_str, diag in rd.items():
                rows.append({
                    'tau': tau, 'seed': seed, 'f': int(f_str),
                    'S_star_size': len(diag['S_star']),
                    'P': d.get('n_parties_total'),
                    'Z_f': diag['Z_f'],
                    'eps': diag['eps'],
                    'eps_sum': diag['eps_sum_over_Sstar'],
                    'residual_term': diag['residual_term_outside_S'],
                    'bound': diag['bound_rhs'],
                    'bound_post': diag.get('bound_rhs_aposteriori', diag['bound_rhs']),
                    'R_f': diag['R_f_measured_abs'],
                    'R_pre': diag.get('R_f_preScrub_abs', float('nan')),
                    'holds': diag['bound_holds'],
                    'apriori_holds': diag.get('apriori_bound_holds', diag['bound_holds']),
                    'tight': diag['bound_tightness'],
                })
    return rows


def agg(rows, keys, vals):
    """mean/std of vals grouped by keys."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = {}
    for gk, rs in groups.items():
        out[gk] = {v: (float(np.mean([r[v] for r in rs])),
                       float(np.std([r[v] for r in rs]))) for v in vals}
        out[gk]['n'] = len(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./results/mfeat_tau')
    ap.add_argument('--out', default='./figs')
    ap.add_argument('--sweep', default='tau', choices=['tau', 'rho'],
                    help="swept variable / dir prefix: tau_* (mfeat) or rho_* (synthvfl)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    xlabel = (r'ownership coverage $\tau_{\mathrm{own}}$' if args.sweep == 'tau'
              else r'specialization $\rho$')

    rows = load_runs(args.root, sweep=args.sweep)
    if not rows:
        print(f"NO residual_diag found under {args.root} — is the new roar run done?")
        return
    classes = sorted(set(r['f'] for r in rows))
    taus = sorted(set(r['tau'] for r in rows))
    P = rows[0]['P']
    print(f"loaded {len(rows)} rows | classes={classes} | taus={taus} | P={P}")

    vals = ['S_star_size', 'Z_f', 'eps', 'eps_sum', 'residual_term',
            'bound', 'bound_post', 'R_f', 'R_pre', 'tight']
    A = agg(rows, ['f', 'tau'], vals)

    # text table + sanity
    n_violations = sum(1 for r in rows if not r['holds'])
    n_apriori_viol = sum(1 for r in rows if not r['apriori_holds'])
    print(f"\nbound violations (a-posteriori, MUST be 0): {n_violations}/{len(rows)}")
    print(f"bound violations (a-priori Thm1):           {n_apriori_viol}/{len(rows)}")
    print(f"\n{'f':>2} {'tau':>4} {'|S*|/P':>7} {'E|R_f|':>8} {'bound':>8} "
          f"{'resid':>7} {'eps':>6} {'tight':>6} {'n':>3}")
    summary = {}
    for f in classes:
        for tau in taus:
            g = A.get((f, tau))
            if not g:
                continue
            sstar = g['S_star_size'][0]
            print(f"{f:>2} {tau:>4.2f} {sstar/P:>7.2f} {g['R_f'][0]:>8.4f} "
                  f"{g['bound'][0]:>8.4f} {g['residual_term'][0]:>7.4f} "
                  f"{g['eps'][0]:>6.3f} {g['tight'][0]:>6.2f} {g['n']:>3}")
            summary[f"f{f}_tau{tau}"] = {k: g[k] for k in vals} | {
                'S_star_frac': sstar / P, 'n': g['n']}

    with open(os.path.join(args.out, 'fig1_summary.json'), 'w') as fh:
        json.dump({'P': P, 'classes': classes, 'taus': taus,
                   'aposteriori_violations': n_violations,
                   'apriori_violations': n_apriori_viol,
                   'cells': summary}, fh, indent=2)

    # ---- plotting ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — wrote fig1_summary.json only")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap('tab10')

    # Panel (a): measured vs bound vs residual term across tau, per class
    axa = axes[0]
    for i, f in enumerate(classes):
        c = cmap(i)
        xs = [tau for tau in taus if (f, tau) in A]
        meas = [A[(f, tau)]['R_f'][0] for tau in xs]
        meas_e = [A[(f, tau)]['R_f'][1] for tau in xs]
        bnd = [A[(f, tau)]['bound'][0] for tau in xs]
        res = [A[(f, tau)]['residual_term'][0] for tau in xs]
        axa.errorbar(xs, meas, yerr=meas_e, marker='o', color=c,
                     label=f'class {f}: E|R_f|', lw=2, capsize=3)
        axa.plot(xs, bnd, '--', color=c, alpha=0.8, label=f'class {f}: Thm-1 bound')
        axa.plot(xs, res, ':', color=c, alpha=0.6, label=f'class {f}: (1-τ)Z_f')
    axa.set_xlabel(xlabel)
    axa.set_ylabel(r'residual class-$f$ evidence  $\mathbb{E}|R_f|$')
    axa.set_title('(a) Theorem-1 residual-leakage bound')
    axa.legend(fontsize=6, ncol=1, loc='best')
    axa.grid(alpha=0.3)

    # Panel (b): certified comm-vs-leakage Pareto
    axb = axes[1]
    for i, f in enumerate(classes):
        c = cmap(i)
        pts = sorted([(A[(f, tau)]['S_star_size'][0] / P,
                       A[(f, tau)]['R_f'][0], tau) for tau in taus if (f, tau) in A])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        axb.plot(xs, ys, marker='s', color=c, lw=2, label=f'class {f}')
        for x, y, tau in pts:
            axb.annotate(f'{tau:.1f}', (x, y), fontsize=6, alpha=0.7,
                         textcoords='offset points', xytext=(3, 3))
    axb.set_xlabel(r'communication  $|S^*|/P$')
    axb.set_ylabel(r'residual leakage  $\mathbb{E}|R_f|$')
    axb.set_title('(b) Certified comm–leakage frontier (τ labels)')
    axb.legend(fontsize=7)
    axb.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(args.out, 'fig1_residual_bound.png')
    fig.savefig(png, dpi=160, bbox_inches='tight')
    print(f"\nwrote {png}")
    print(f"wrote {os.path.join(args.out, 'fig1_summary.json')}")


if __name__ == '__main__':
    main()
