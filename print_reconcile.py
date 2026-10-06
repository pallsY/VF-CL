"""Reconcile audit verdicts against EMPIRICAL pure-CL trajectories.
For each method in results/_clvalidate/<m>/, read the per-task trajectory and
classify by LEARNING PROCESS (not final metric):
  fresh_t  = acc of task t at the event right after it is trained (diagonal)
  final_t  = acc of task t at the last event
  avg_fresh        = mean fresh_t over all tasks         (can it learn new tasks?)
  retain_old       = mean final_t over all-but-last task (does it keep old tasks?)
Classification:
  avg_fresh < 0.5                  -> 'CANT-LEARN-NEW'   (新任务学不动)
  retain_old < 0.05                -> 'FORGETS-ALL'      (旧任务全忘)
  retain_old < 0.25                -> 'partial-forget'
  else                             -> 'HEALTHY'
"""
import json, glob, os

AUDIT = {  # from the 26-agent code audit (lwf_fim/prl corrected after manual re-check)
    'finetune': 'valid(lower-bound)', 'lwf': 'valid', 'lwf_fim': 'valid(audit-FP)',
    'lwf_wa': 'valid(fixed)', 'ewc': 'valid', 'gpm': 'valid', 'proto_aug': 'valid',
    'proto_fedspace': 'WEAK', 'adagauss': 'valid', 'afc': 'valid', 'prl': 'valid(fixed)',
    'target': 'valid', 'er_ace': 'WEAK(no-buf)', 'er': 'valid(exemplar)',
    'der_pp': 'valid(exemplar)', 'proto_evolve': 'valid(KD-carries)',
}
ORDER = ['finetune', 'lwf', 'lwf_fim', 'lwf_wa', 'ewc', 'gpm', 'proto_aug',
         'proto_fedspace', 'adagauss', 'afc', 'prl', 'target', 'er_ace', 'er',
         'der_pp', 'proto_evolve']


def analyze(f):
    d = json.load(open(f))
    h = d['task_acc_history']
    cil = [e for e in h if e['step'].endswith('CIL')]
    tasks = sorted({k for e in cil for k in e['per_task_accs']},
                   key=lambda s: int(s.split('_')[-1]))
    fresh, final = {}, {}
    last = cil[-1]['per_task_accs']
    for i, t in enumerate(tasks):
        # task t is fresh at the i-th CIL event (no UL in this sweep)
        if i < len(cil) and t in cil[i]['per_task_accs']:
            fresh[t] = cil[i]['per_task_accs'][t]
        final[t] = last.get(t, 0.0)
    avg_fresh = sum(fresh.values()) / max(len(fresh), 1)
    old = [final[t] for t in tasks[:-1]]
    retain_old = sum(old) / max(len(old), 1) if old else 0.0
    if avg_fresh < 0.5:
        cls = 'CANT-LEARN-NEW'
    elif retain_old < 0.05:
        cls = 'FORGETS-ALL'
    elif retain_old < 0.25:
        cls = 'partial-forget'
    else:
        cls = 'HEALTHY'
    return d['cl_metrics']['AA_final'], avg_fresh, retain_old, cls


print('%-15s %-9s | %7s %7s %8s | %-16s | %s' %
      ('method', 'AA_fin', 'fresh', 'retain', '', 'EMPIRICAL', 'AUDIT'))
print('-' * 92)
for m in ORDER:
    fs = glob.glob('results/_clvalidate/%s/**/results.json' % m, recursive=True)
    if not fs:
        print('%-15s %-9s | %43s | %s' % (m, '(pending)', '', AUDIT.get(m, '?')))
        continue
    aa, af, ro, cls = analyze(sorted(fs)[0])
    print('%-15s %-9.3f | %7.2f %7.2f %8s | %-16s | %s' %
          (m, aa, af, ro, '', cls, AUDIT.get(m, '?')))
