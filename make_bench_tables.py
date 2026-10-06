"""Build the 3 benchmark tables from results/bench (dispatch_bench.sh output).

Table 1 (CL axis):  per dataset, per CL method -> AA_cil, BWT  (continual quality).
Table 2 (UL axis):  per dataset, per UL method -> forget, retain, parties_touched,
                    relearn-AUC (vs the dataset's retrain-oracle floor), MIA.
Table 3 (UL x CL):  parties_touched of the unlearn for each CL backbone (ours vs
                    baselines) -> the unified contribution (our CL -> cheaper UL).
"""
import glob, json, os
from collections import defaultdict
import numpy as np

DSETS = ['mfeat', 'cov', 'har']
CL = ['finetune', 'ewc', 'lwf', 'lwf_wa', 'er', 'der_pp', 'afc', 'adagauss', 'proto_evolve', 'ours']
UL = ['retrain', 'gradient_ascent', 'luv', 'fucrt', 'fedup', 'fedosd', 'fedau', 'radapt_router', 'roar']


def load(pattern):
    """mean/std over seeds for one cell; returns dict of metric->(mean,std)."""
    rows = []
    for p in glob.glob(pattern + '/**/results.json', recursive=True):
        d = json.load(open(p))
        cl = d.get('cl_metrics', {}) or {}
        ulm = (d.get('ul_metrics') or [{}])[-1]
        rl = ulm.get('relearn_auc', {})
        rows.append({
            'AA': cl.get('AA_cil', np.nan),
            'AAdeb': cl.get('AA_cil_debiased', np.nan),
            'AAtil': cl.get('AA_cil_taskil', np.nan), 'BWT': cl.get('BWT', np.nan),
            'forget': ulm.get('forget_acc', np.nan), 'retain': ulm.get('retain_acc', np.nan),
            'touched': ulm.get('parties_touched', np.nan), 'mia': ulm.get('mia_score', np.nan),
            'auc': (list(rl.values())[0] if rl else np.nan),
        })
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        vs = [r[k] for r in rows if r[k] == r[k]]
        out[k] = (float(np.mean(vs)), float(np.std(vs)), len(vs)) if vs else (np.nan, 0, 0)
    return out


def oracle(ds):
    fs = glob.glob(f'results/bench/oracle_{ds}/**/attack_baseline.json', recursive=True)
    return json.load(open(fs[0]))['oracle_relearn_auc'] if fs else np.nan


def main():
    print('=' * 78)
    print('TABLE 1 - CL quality. 3 fair readouts: raw class-IL / debiased / task-IL')
    print('(raw class-IL is recency-biased -> unfair to ALL exemplar-free methods;')
    print(' task-IL shows whether representation is preserved.)')
    print('=' * 78)
    for ds in DSETS:
        print(f'\n--- {ds} ---')
        print(f'{"CL method":>14} {"rawAA":>7} {"debiasAA":>9} {"taskIL":>7} {"BWT":>8}')
        for cl in CL:
            c = load(f'results/bench/A_{ds}_cl-{cl}')
            if not c:
                print(f'{cl:>14} {"-":>7}'); continue
            print(f'{cl:>14} {c["AA"][0]:>7.3f} {c["AAdeb"][0]:>9.3f} {c["AAtil"][0]:>7.3f} {c["BWT"][0]:>8.3f}')

    print('\n' + '=' * 70)
    print('TABLE 2 - UL comparison (CL fixed = finetune). forget/retain/touched/AUC')
    print('oracle floor (relearn AUC of retrain-from-scratch):',
          {d: round(oracle(d), 3) for d in DSETS})
    print('=' * 70)
    for ds in DSETS:
        print(f'\n--- {ds} (oracle AUC={oracle(ds):.3f}) ---')
        print(f'{"UL method":>16} {"forget":>7} {"retain":>7} {"touched":>8} {"relearnAUC":>11} {"MIA":>6}')
        for ul in UL:
            c = load(f'results/bench/B_{ds}_ul-{ul}')
            if not c:
                print(f'{ul:>16} {"(missing)":>7}'); continue
            tag = ' <- ours' if ul == 'roar' else ''
            print(f'{ul:>16} {c["forget"][0]:>7.3f} {c["retain"][0]:>7.3f} '
                  f'{c["touched"][0]:>6.1f}/6 {c["auc"][0]:>11.3f} {c["mia"][0]:>6.3f}{tag}')

    print('\n' + '=' * 70)
    print('TABLE 3 - UL x CL: parties touched by the unlearn, per CL backbone')
    print('(our localized-retrain UL on each CL; lower = cheaper unlearning)')
    print('=' * 70)
    print(f'{"CL backbone":>14}' + ''.join(f'{d:>10}' for d in DSETS))
    for cl in CL:
        line = f'{cl:>14}'
        for ds in DSETS:
            c = load(f'results/bench/A_{ds}_cl-{cl}')
            line += f'{(("%.1f"%c["touched"][0]) if c else "-"):>10}'
        print(line)
    print('\n(ours = finetune + ownership-concentration regularizer)')


if __name__ == '__main__':
    main()
