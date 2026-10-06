"""Print the per-task acc trajectory matrix from a results.json (task_acc_history)."""
import json, glob, sys

f = sorted(glob.glob(sys.argv[1]))[-1]
d = json.load(open(f))
tah = d['task_acc_history']
cm = d.get('cl_metrics', {})
ntasks = max((len(e['per_task_accs']) for e in tah), default=0)

print('file:', f)
cfg = d.get('config', {})
print('config:', {k: cfg.get(k) for k in ['cl_method', 'num_parties', 'aggregation', 'cosine_head', 'epochs_per_task']})
print()
hdr = ' '.join(f'  t{i} ' for i in range(ntasks))
print(f"{'after':>12} |{hdr}| overall")
print('-' * (15 + 7 * ntasks + 10))
for e in tah:
    pa = e['per_task_accs']
    cells = []
    for i in range(ntasks):
        k = f'task_{i}'
        cells.append(f'{pa[k]:.3f}' if k in pa else '  -  ')
    row = ' '.join(f'{c:>5}' for c in cells)
    print(f"{e['step']:>12} | {row} | {e['overall_acc']:.4f}")
print()
print('cl_metrics:', {k: cm.get(k) for k in ['AA_cil', 'AA_final', 'BWT']})
