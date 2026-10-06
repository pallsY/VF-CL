"""Publication LaTeX (booktabs) for the 3 benchmark tables, with mean$\\pm$std over
3 seeds and best(\\textbf)/second(\\underline) highlighting among comparable methods.
ewc=lambda5000 (bench2); ours=proto_evolve+ownership-concentration (bench2);
target/joint (bench3). Raw-sample replay (ER/DER++) excluded on principle.
"""
import glob, json
import numpy as np

DS = ['mfeat', 'cov', 'har']
DSN = {'mfeat': 'mfeat', 'cov': 'Covertype', 'har': 'HAR'}


def cell(pattern):
    keys = ['AA', 'AAt', 'BWT', 'forget', 'retain', 'touched', 'auc']
    acc = {k: [] for k in keys}
    for p in glob.glob(pattern + '/**/results.json', recursive=True):
        d = json.load(open(p)); cl = d.get('cl_metrics', {}); ulm = (d.get('ul_metrics') or [{}])[-1]
        src = {'AA': cl.get('AA_cil'), 'AAt': cl.get('AA_cil_taskil'), 'BWT': cl.get('BWT'),
               'forget': ulm.get('forget_acc'), 'retain': ulm.get('retain_acc'),
               'touched': ulm.get('parties_touched'),
               'auc': (list(ulm.get('relearn_auc', {}).values())[0] if ulm.get('relearn_auc') else None)}
        for k in keys:
            if src.get(k) is not None:
                acc[k].append(src[k])
    return {k: (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan) for k, v in acc.items()}


def A(ds, cl):
    return {'ewc': f'results/bench2/A_{ds}_cl-ewc5k', 'ours': f'results/bench2/A_{ds}_cl-ours2',
            'target': f'results/bench3/A_{ds}_cl-target', 'joint': f'results/bench3/joint_{ds}'
            }.get(cl, f'results/bench/A_{ds}_cl-{cl}')


def hl(ms, direction, rankable, dec=3, std=True, second=True):
    """ms: list of (mean,std). rankable: bool list (compete for bold/underline).
    second=False -> only bold the best (for count columns with big ties)."""
    cand = sorted({round(ms[i][0], 6) for i in range(len(ms))
                   if rankable[i] and ms[i][0] == ms[i][0]}, reverse=(direction == 'max'))
    best = cand[0] if cand else None
    second = (cand[1] if len(cand) > 1 else None) if second else None
    out = []
    for i, (m, s) in enumerate(ms):
        if m != m:
            out.append('--'); continue
        b = f'{m:.{dec}f}'
        if rankable[i] and best is not None and round(m, 6) == best:
            b = f'\\textbf{{{b}}}'
        elif rankable[i] and second is not None and round(m, 6) == second:
            b = f'\\underline{{{b}}}'
        out.append(b + (f'$_{{\\pm{s:.2f}}}$' if (std and s == s) else ''))
    return out


# ================= Table 1: CL quality =================
CL = [('finetune', 'Finetune (lower bound)', False), ('ewc', 'EWC', True), ('lwf', 'LwF', True),
      ('lwf_wa', 'LwF-WA', True), ('afc', 'AFC', True), ('adagauss', 'AdaGauss$^{g}$', True),
      ('target', 'TARGET$^{g}$', True), ('proto_evolve', 'Proto-Evolve', True),
      ('ours', '\\textbf{Ours (Proto+OC)}', True), ('joint', 'Joint (upper bound)', False)]
C = {ds: {cl: cell(A(ds, cl)) for cl, _, _ in CL} for ds in DS}
rank = [r for _, _, r in CL]
print('% ===== Table 1: Continual-learning quality =====')
print('\\begin{table*}[t]\\centering\\small')
print('\\caption{Continual-learning quality on three VFL datasets (mean$\\pm$std, 3 seeds). '
      'AA = class-incremental accuracy; T-IL = task-incremental accuracy; '
      'BWT = backward transfer (forgetting; closer to 0 better). \\textbf{Bold}/\\underline{underline} '
      '= best/second among comparable methods. All methods are raw-exemplar-free (no stored old '
      'samples), as required by unlearning; $^{g}$=generative replay. Raw-sample replay '
      '(ER/DER++) is excluded: it retains the very data unlearning must delete.}')
print('\\begin{tabular}{l' + 'ccc' * len(DS) + '}\\toprule')
print(' & ' + ' & '.join('\\multicolumn{3}{c}{%s}' % DSN[d] for d in DS) + ' \\\\')
print(''.join('\\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(DS))))
print('Method & ' + ' & '.join('AA & T-IL & BWT' for _ in DS) + ' \\\\\\midrule')
cols = {}
for d in DS:
    for m in ['AA', 'AAt', 'BWT']:
        cols[(d, m)] = hl([C[d][cl][m] for cl, _, _ in CL], 'max', rank)
for ri, (cl, name, _) in enumerate(CL):
    row = [name] + [cols[(d, m)][ri] for d in DS for m in ['AA', 'AAt', 'BWT']]
    print(' & '.join(row) + ' \\\\')
    if cl == 'finetune':
        print('\\midrule')
    if cl == 'ours':
        print('\\midrule')
print('\\bottomrule\\end{tabular}\\end{table*}\n')

# ================= Table 2: UL comparison =================
UL = [('retrain', 'Retrain (oracle)', False), ('gradient_ascent', 'GradAscent', True),
      ('luv', 'LUV', True), ('fucrt', 'FUCRT', True), ('fedup', 'FedUP', True),
      ('fedau', 'FedAU', True), ('fedosd', 'FedOSD', True), ('radapt_router', 'RAdapt', True),
      ('roar', '\\textbf{Ours (local retrain)}', True)]
U = {ds: {ul: cell(f'results/bench/B_{ds}_ul-{ul}') for ul, _, _ in UL} for ds in DS}
rankU = [r for _, _, r in UL]
print('% ===== Table 2: Unlearning comparison (CL backbone = Finetune) =====')
print('\\begin{table*}[t]\\centering\\small')
print('\\caption{Class-unlearning on three VFL datasets ($P{=}6$ parties, mean$\\pm$std, 3 seeds). '
      'Ret = retain acc ($\\uparrow$); \\#P = parties whose encoder is modified '
      '(communication, $\\downarrow$); Fgt = forget-class acc ($\\downarrow$). '
      '\\textbf{Bold}/\\underline{underline} = best/second (excl.\\ oracle). Ours matches the '
      'best-quality methods at a fraction of the communication.}')
print('\\begin{tabular}{l' + 'ccc' * len(DS) + '}\\toprule')
print(' & ' + ' & '.join('\\multicolumn{3}{c}{%s}' % DSN[d] for d in DS) + ' \\\\')
print(''.join('\\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(DS))))
print('Method & ' + ' & '.join('Ret & \\#P & Fgt' for _ in DS) + ' \\\\\\midrule')
rcol = {d: hl([U[d][ul]['retain'] for ul, _, _ in UL], 'max', rankU) for d in DS}
# #P ranking excludes GradAscent: its 0/6 is head-only (no bottom scrub) and it
# fails to unlearn (high Fgt), so its low comm is not a valid win.
rankP = [r and ul != 'gradient_ascent' for ul, _, r in UL]
pcol = {d: hl([U[d][ul]['touched'] for ul, _, _ in UL], 'min', rankP, dec=0, std=False, second=False) for d in DS}
for ri, (ul, name, _) in enumerate(UL):
    row = [name]
    for d in DS:
        p = pcol[d][ri]
        p = p if p == '--' else p + '/6'
        fg = U[d][ul]['forget'][0]
        row += [rcol[d][ri], p, ('--' if fg != fg else f'{fg:.2f}')]
    print(' & '.join(row) + ' \\\\')
    if ul == 'retrain':
        print('\\midrule')
print('\\bottomrule\\end{tabular}\\end{table*}\n')

# ================= Table 3: UL x CL =================
print('% ===== Table 3: UL x CL (communication of unlearning per CL backbone) =====')
print('\\begin{table}[t]\\centering\\small')
print('\\caption{Parties touched by our localized-retrain unlearning under each CL backbone '
      '(mean over 3 seeds; $\\downarrow$ cheaper). Ownership-aware CL (Ours) concentrates each '
      'class so unlearning touches the fewest parties. \\textbf{Bold}=fewest.}')
print('\\begin{tabular}{l' + 'c' * len(DS) + '}\\toprule')
print('CL backbone & ' + ' & '.join(DSN[d] for d in DS) + ' \\\\\\midrule')
CL3 = [(cl, name) for cl, name, _ in CL if cl not in ('joint',)]
tcol = {d: hl([C[d][cl]['touched'] for cl, _ in CL3], 'min', [True]*len(CL3), dec=1, std=False, second=False) for d in DS}
for ri, (cl, name) in enumerate(CL3):
    print(' & '.join([name] + [tcol[d][ri] for d in DS]) + ' \\\\')
    if cl == 'finetune':
        print('\\midrule')
print('\\bottomrule\\end{tabular}\\end{table}')
