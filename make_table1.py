"""Emit main Table 1 (VFL-CLU benchmark, Order 1) as LaTeX.

Layout per the aligned paper structure: rows = CL x UL combos (CL | UL columns,
grouped by CL backbone), column groups = 3 datasets x (RA UA MIA #P), all
END-OF-STREAM metrics (relapse-inclusive). mean +- std over 3 seeds; std as
subscript. Missing / known-invalid cells print '-'.
Usage: python make_table1.py > table1_main.tex
"""
import glob, json, sys
import numpy as np
from collect_grid import best_multiseed_dir

CL_ROWS = [
    ('finetune', 'Finetune'),
    ('ewc', 'EWC'),
    ('lwf', 'LwF'),
    ('lwf_wa', 'LwF-WA'),
    ('lwf_fim', 'LwF-FIM'),
    ('afc', 'AFC'),
    ('gpm', 'GPM'),
    ('adagauss', 'AdaGauss$^{g}$'),
    ('target', 'TARGET$^{g}$'),
    ('proto_evolve', 'Proto-Evolve'),
    ('proto_fedspace', 'Proto-FedSpace'),
    ('ours_protoOC', r'\textbf{Proto+OC (ours)}'),
]
UL_ROWS = [
    ('retrain', 'Retrain (oracle)'),
    ('ga', 'GradAscent'),
    ('luv', 'LUV'),
    ('mode', 'MoDe'),
    ('fucrt', 'FUCRT'),
    ('fedup', 'FedUP'),
    ('fedosd', 'FedOSD'),
    ('fedau', 'FedAU'),
    ('ours_localrt', r'\textbf{LocalRT (ours)}'),
]
DATASETS = [
    ('cifar', 'results/grid_cifar/o1', 'CIFAR-100', 4),
    ('covtype', 'results/grid_tab/covtype_o1', 'Covertype', 6),
    ('nuswide', 'results/grid_tab/nuswide_o1', 'NUS-WIDE', 6),
]
# cells known-invalid pending fixes (adagauss eval clash; ewc x cifar collapse)
INVALID = {('cifar', 'ewc'), ('cifar', 'adagauss'),
           ('covtype', 'adagauss'), ('nuswide', 'adagauss')}


KL = json.load(open('kl_posthoc.json')) if __import__('os').path.exists('kl_posthoc.json') else {}


def cell_metrics(root, cl, ul):
    _, seed_files = best_multiseed_dir(f'{root}/{cl}__{ul}')
    if len(seed_files) < 3:
        return None
    ra, ua, mia, pt, rte = [], [], [], [], []
    for sf in sorted(seed_files)[-3:]:
        d = json.load(open(sf))
        fe = d.get('final_ul_eval') or {}
        ra.append(fe.get('retain_acc'))
        ua.append(fe.get('forget_acc'))
        auc = fe.get('relearn_auc_final') or {}
        if auc:
            mia.append(float(np.mean(list(auc.values()))))
        ulm = d.get('ul_metrics') or []
        ev = [u['parties_touched'] for u in ulm if u.get('parties_touched') is not None]
        if ev:
            pt.append(float(np.mean(ev)))
        rte.append(sum(t['time_seconds'] for t in d.get('timing', []) if 'UL' in t['step']))
    kl = [float(v) for v in KL.get(f'{root}/{cl}__{ul}', {}).values()]
    def ms(v):
        v = [x for x in v if x is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else None
    return {'RA': ms(ra), 'UA': ms(ua), 'MIA': ms(mia), 'P': ms(pt),
            'KL': ms(kl), 'RTE': ms(rte)}


def fmt(m, key, prec=2):
    if not m or not m.get(key):
        return '-'
    mu, sd = m[key]
    if key == 'P':
        return f'{mu:.1f}'
    if key == 'RTE':
        return f'{mu:.0f}'
    if key == 'MIAx':
        return f'{mu:+.3f}'
    return f'{mu:.{prec}f}$_{{\\pm{sd:.2f}}}$'


def main():
    out = []
    out.append(r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3.2pt}')
    out.append(r'\caption{\textbf{Main benchmark: task-agnostic VFL-CLU} on three datasets')
    out.append(r'(standard order; mean$_{\pm\text{std}}$, 3 seeds). Each row is a CL$\times$UL')
    out.append(r'combination run on the interleaved learn/forget queue. RA$\uparrow$ =')
    out.append(r'retained-class accuracy at stream end; UA$\downarrow$ = forgotten-class')
    out.append(r'accuracy at stream end (includes relapse from post-unlearning training);')
    out.append(r'MIA$_{\Delta}\!\downarrow$ = re-learned linear-head ROC-AUC minus the same-row')
    out.append(r'retrain oracle (intrinsic-separability floor; $\approx$0 = leaks no more than')
    out.append(r'full retraining); KL$\downarrow$ = mean KL divergence to the same-row oracle')
    out.append(r'on the retained-class support (0 for the oracle by definition);')
    out.append(r'\#P$\downarrow$ = parties whose encoder is modified per')
    out.append(r'unlearning event (of 4 / 6 / 6); RTE$\downarrow$ = unlearning wall-clock (s).')
    out.append(r'$^{g}$generative or prototype replay')
    out.append(r"(raw-exemplar replay is excluded by protocol). ``-'' = pending.}")
    out.append(r'\label{tab:main}')
    out.append(r'\begin{tabular}{ll|cccccc|cccccc|cccccc}\toprule')
    out.append(r' & & \multicolumn{6}{c|}{CIFAR-100 ($P{=}4$)} & \multicolumn{6}{c|}{Covertype ($P{=}6$)} & \multicolumn{6}{c}{NUS-WIDE ($P{=}6$)}\\')
    hdr = r'CL & UL'
    for _ in range(3):
        hdr += r' & RA$\uparrow$ & UA$\downarrow$ & MIA$_{\Delta}\!\downarrow$ & KL$\downarrow$ & \#P$\downarrow$ & RTE$\downarrow$'
    out.append(hdr + r'\\\midrule')

    for ci, (cl, cl_disp) in enumerate(CL_ROWS):
        # per-row oracle AUC floor (same CL x retrain) for the MIA-excess column
        floors = {}
        for ds, root, _, _ in DATASETS:
            om = None if (ds, cl) in INVALID else cell_metrics(root, cl, 'retrain')
            floors[ds] = om['MIA'][0] if om and om.get('MIA') else None
        for ui, (ul, ul_disp) in enumerate(UL_ROWS):
            row = (cl_disp if ui == 0 else '') + ' & ' + ul_disp
            for ds, root, _, _ in DATASETS:
                m = None if (ds, cl) in INVALID else cell_metrics(root, cl, ul)
                if m and m.get('MIA') and floors[ds] is not None:
                    m = dict(m, MIAx=(m['MIA'][0] - floors[ds], 0.0))
                row += (f' & {fmt(m,"RA")} & {fmt(m,"UA")} & {fmt(m,"MIAx")}'
                        f' & {fmt(m,"KL")} & {fmt(m,"P")} & {fmt(m,"RTE")}')
            out.append(row + r'\\')
        if ci < len(CL_ROWS) - 1:
            out.append(r'\midrule')
    out.append(r'\bottomrule\end{tabular}\end{table*}')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
