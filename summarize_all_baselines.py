"""Aggregate EVERY (cl_method x ul_method x dataset) combination found under
results/ into one master table. For each combo keeps the most recent run and
reports 3-seed-mean metrics. Pure data aggregation — no interpretation."""
import json, glob, os
from collections import defaultdict


def g(d, sect, key):
    v = d.get(sect, {})
    if isinstance(v, dict):
        x = v.get(key)
        if isinstance(x, dict):
            return x.get('mean')
        return x
    return None


def load(f):
    try:
        d = json.load(open(f))
    except Exception:
        return None
    if 'cl_metrics' not in d:
        return None
    return {
        'cl': d.get('cl_method'), 'ul': d.get('ul_method'),
        'data': (d.get('config', {}) or {}).get('data') or _infer_data(f),
        'AA_final': g(d, 'cl_metrics', 'AA_final'),
        'BWT': g(d, 'cl_metrics', 'BWT'),
        'forget': g(d, 'ul_metrics', 'forget_acc'),
        'retain': g(d, 'ul_metrics', 'retain_acc'),
        'mia': g(d, 'ul_metrics', 'mia_score'),
        'mtime': os.path.getmtime(f), 'path': f,
    }


def _infer_data(f):
    for tok in ('cifar100', 'cifar10', 'tinyimagenet'):
        if tok in f:
            return tok
    return '?'


# keep most-recent run per (cl, ul, data, top-group)
best = {}
for f in glob.glob('results/**/aggregated.json', recursive=True):
    r = load(f)
    if not r or not r['cl']:
        continue
    grp = f.split('/')[1] if f.startswith('results/') else 'x'  # frag / full_benchmark_* / combo dir / _diag
    grp = 'frag' if grp == 'frag' else ('full_benchmark' if grp.startswith('full_benchmark') else
                                        ('DIAG' if grp.startswith('_') else 'standalone'))
    key = (r['data'], r['cl'], r['ul'], grp)
    if key not in best or r['mtime'] > best[key]['mtime']:
        best[key] = r


def fmt(v, p=3):
    return ('%.*f' % (p, v)) if isinstance(v, (int, float)) else '  -  '


for grp in ['frag', 'full_benchmark', 'standalone', 'DIAG']:
    rows = [r for k, r in best.items() if k[3] == grp]
    if not rows:
        continue
    print('\n' + '=' * 96)
    print('GROUP: %s   (%d combos)' % (grp, len(rows)))
    print('=' * 96)
    print('%-9s %-14s %-16s | %7s %7s | %6s %6s %5s' %
          ('data', 'cl', 'ul', 'AA_fin', 'BWT', 'forget', 'retain', 'mia'))
    print('-' * 96)
    for r in sorted(rows, key=lambda x: (x['data'], x['cl'], x['ul'])):
        print('%-9s %-14s %-16s | %7s %7s | %6s %6s %5s' % (
            r['data'], r['cl'], r['ul'], fmt(r['AA_final']), fmt(r['BWT']),
            fmt(r['forget'], 2), fmt(r['retain'], 2), fmt(r['mia'], 2)))

print('\nTOTAL distinct combos:', len(best))
