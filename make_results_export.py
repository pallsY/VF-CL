"""Export ALL benchmark results to one Excel workbook + flat CSVs.

Sheets:
  README          - column definitions + provenance
  main_grid       - one row per (dataset, order, CL, UL): mean/std over seeds
  main_grid_seeds - one row per individual seed run (re-aggregate as you like)
  appendix_clonly - CL methods, pure continual queue (no UL events)
  appendix_ulonly - UL methods, joint backbone + one-shot forget
CSVs: results_export/vfclu_main_grid.csv, results_export/vfclu_all_runs.csv
Regenerate any time: python make_results_export.py
"""
import glob, json, os
import numpy as np
import pandas as pd
from collect_grid import best_multiseed_dir

KL = json.load(open('kl_posthoc.json')) if os.path.exists('kl_posthoc.json') else {}
# rows known-invalid pending method fixes
INVALID = {('cifar100', 'ewc'), ('cifar100', 'adagauss'),
           ('covtype', 'adagauss'), ('nuswide', 'adagauss')}


def seed_rows(combo_dir, meta):
    _, seed_files = best_multiseed_dir(combo_dir)
    rows = []
    for sf in sorted(seed_files):
        d = json.load(open(sf))
        seed = d.get('config', {}).get('seed')
        cm = d.get('cl_metrics') or {}
        fe = d.get('final_ul_eval') or {}
        ulm = d.get('ul_metrics') or []
        pt = [u['parties_touched'] for u in ulm if u.get('parties_touched') is not None]
        auc = fe.get('relearn_auc_final') or {}
        ua_ev = [u.get('forget_acc') for u in ulm if u.get('forget_acc') is not None]
        kl_v = KL.get(combo_dir, {}).get(str(seed))
        rows.append(dict(
            meta, seed=seed,
            RA=fe.get('retain_acc'), UA_end=fe.get('forget_acc'),
            UA_after_event=(float(np.mean(ua_ev)) if ua_ev else None),
            MIA_sample=fe.get('mia_score'),
            relearn_AUC=(float(np.mean(list(auc.values()))) if auc else None),
            KL_to_oracle=kl_v,
            parties_touched=(float(np.mean(pt)) if pt else None),
            RTE_ul_s=sum(t['time_seconds'] for t in d.get('timing', []) if 'UL' in t['step']),
            RTE_total_s=sum(t['time_seconds'] for t in d.get('timing', [])),
            AA_final=cm.get('AA_final'), AA_cil=cm.get('AA_cil'), BWT=cm.get('BWT'),
            valid=int((meta['dataset'], meta['CL']) not in INVALID),
        ))
    return rows


def collect_main():
    rows = []
    specs = [('results/grid_tab/covtype_o1', 'covtype', 1), ('results/grid_tab/covtype_o2', 'covtype', 2),
             ('results/grid_tab/nuswide_o1', 'nuswide', 1), ('results/grid_tab/nuswide_o2', 'nuswide', 2),
             ('results/grid_cifar/o1', 'cifar100', 1), ('results/grid_cifar/o2', 'cifar100', 2)]
    for root, ds, od in specs:
        for combo_dir in sorted(glob.glob(f'{root}/*__*')):
            cl, ul = os.path.basename(combo_dir).split('__', 1)
            rows += seed_rows(combo_dir, dict(suite='main_grid', dataset=ds, order=od, CL=cl, UL=ul))
    return pd.DataFrame(rows)


def collect_appendix():
    cl_rows, ul_rows = [], []
    for d in sorted(glob.glob('results/appendix/clonly_*/*')):
        ds = d.split('clonly_')[1].split('/')[0]
        cl_rows += seed_rows(d, dict(suite='appendix_clonly', dataset=ds, order=0,
                                     CL=os.path.basename(d), UL='none'))
    for d in sorted(glob.glob('results/appendix/ulonly_*/*')):
        ds = d.split('ulonly_')[1].split('/')[0]
        ul_rows += seed_rows(d, dict(suite='appendix_ulonly', dataset=ds, order=0,
                                     CL='finetune_joint', UL=os.path.basename(d)))
    return pd.DataFrame(cl_rows), pd.DataFrame(ul_rows)


def aggregate(df):
    keys = ['suite', 'dataset', 'order', 'CL', 'UL']
    mets = ['RA', 'UA_end', 'UA_after_event', 'MIA_sample', 'relearn_AUC',
            'KL_to_oracle', 'parties_touched', 'RTE_ul_s', 'RTE_total_s',
            'AA_final', 'AA_cil', 'BWT']
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_seeds=('seed', 'count'), valid=('valid', 'min'),
                **{f'{m}_mean': (m, 'mean') for m in mets},
                **{f'{m}_std': (m, 'std') for m in mets}).reset_index()
    return out.round(4)


def main():
    os.makedirs('results_export', exist_ok=True)
    main_df = collect_main()
    cl_df, ul_df = collect_appendix()
    all_runs = pd.concat([main_df, cl_df, ul_df], ignore_index=True)
    main_agg = aggregate(main_df)
    cl_agg = aggregate(cl_df).dropna(axis=1, how='all')
    ul_agg = aggregate(ul_df).dropna(axis=1, how='all')

    all_runs.to_csv('results_export/vfclu_all_runs.csv', index=False)
    main_agg.to_csv('results_export/vfclu_main_grid.csv', index=False)

    readme = pd.DataFrame([
        ['suite', 'main_grid = CL x UL interleaved queue; appendix_clonly = pure CL (no UL); appendix_ulonly = joint backbone + one-shot forget'],
        ['dataset / order', 'covtype (P=6) / nuswide (P=6) / cifar100 (P=4); order 1 = forget-old, order 2 = forget-recent (appendix)'],
        ['RA', 'retained-class accuracy at stream END'],
        ['UA_end', 'forgotten-class accuracy at stream END (includes relapse)'],
        ['UA_after_event', 'mean forget-class acc right after each UL event (pre-relapse)'],
        ['MIA_sample', 'sample-level entropy membership-inference score (0.5 = chance)'],
        ['relearn_AUC', 're-learned linear-head ROC-AUC on final embeddings; compare to same-row retrain (intrinsic floor), not to 0.5'],
        ['KL_to_oracle', 'mean KL(oracle || method) on retained support vs SAME-ROW retrain, same seed; lower = closer to retrain'],
        ['parties_touched', 'encoders modified per UL event (communication); of P'],
        ['RTE_ul_s / RTE_total_s', 'wall-clock seconds: UL events only / whole stream'],
        ['AA_final / AA_cil / BWT', 'continual-learning metrics from the tracker'],
        ['valid', '0 = known-invalid pending method fix (adagauss all; ewc on cifar100)'],
        ['pending', 'cifar100: lwf_wa/lwf_fim/adagauss rows deferred; ewc row awaiting fix; order 2 not yet run'],
    ], columns=['column', 'definition'])

    with pd.ExcelWriter('results_export/vfclu_benchmark_results.xlsx', engine='openpyxl') as w:
        readme.to_excel(w, sheet_name='README', index=False)
        main_agg.to_excel(w, sheet_name='main_grid', index=False)
        main_df.to_excel(w, sheet_name='main_grid_seeds', index=False)
        cl_agg.to_excel(w, sheet_name='appendix_clonly', index=False)
        ul_agg.to_excel(w, sheet_name='appendix_ulonly', index=False)
        from openpyxl.styles import Font
        for ws in w.book.worksheets:
            for c in ws[1]:
                c.font = Font(name='Arial', bold=True)
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 9), 60)

    print(f'runs: {len(all_runs)}  main combos: {len(main_agg)}  '
          f'clonly: {len(cl_agg)}  ulonly: {len(ul_agg)}')
    print('wrote results_export/vfclu_benchmark_results.xlsx + vfclu_main_grid.csv + vfclu_all_runs.csv')


if __name__ == '__main__':
    main()
