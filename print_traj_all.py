"""Dump RAW per-task trajectories for every method in results/_clvalidate/.
No classification, no judgement — just the numbers, so you can read the
learning process yourself (diagonal = fresh-task acc, columns = retention)."""
import json, glob

ORDER = ['finetune', 'lwf', 'lwf_fim', 'lwf_wa', 'ewc', 'gpm', 'proto_aug',
         'proto_fedspace', 'adagauss', 'afc', 'prl', 'target', 'er_ace', 'er',
         'der_pp', 'proto_evolve']


def dump(m, f):
    d = json.load(open(f))
    cil = [e for e in d['task_acc_history'] if e['step'].endswith('CIL')]
    tasks = sorted({k for e in cil for k in e['per_task_accs']},
                   key=lambda s: int(s.split('_')[-1]))
    print('\n=== %s   AA_final=%.3f   BWT=%.3f ===' % (
        m, d['cl_metrics']['AA_final'], d['cl_metrics'].get('BWT', 0.0)))
    hdr = 'after learning  ' + ''.join('%6s' % t.replace('task_', 't') for t in tasks)
    print(hdr)
    for i, e in enumerate(cil):
        cells = ''.join('%6.2f' % e['per_task_accs'][t] if t in e['per_task_accs']
                        else '    . ' for t in tasks)
        print('  %-13s %s' % ('task%d' % i, cells))


for m in ORDER:
    fs = glob.glob('results/_clvalidate/%s/**/results.json' % m, recursive=True)
    if fs:
        dump(m, sorted(fs)[0])
    else:
        print('\n=== %s   (pending) ===' % m)
