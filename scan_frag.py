"""Scan results/frag/**/aggregated.json into a baseline summary table."""
import json, glob, os, re
import numpy as np

rows = []
for f in glob.glob('results/frag/*/*/**/aggregated.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    # cell = results/frag/<dscell>/<combo>/...
    parts = f.split('/')
    cell = parts[2]            # e.g. c10_4p
    combo = parts[3]           # e.g. proto_evolve_x_luv  or oracle
    cl = d.get('cl_method', combo)
    ul = d.get('ul_method', '-')
    clm = d.get('cl_metrics', {})
    ulm = d.get('ul_metrics', {})

    def g(dct, k):
        v = dct.get(k)
        return v.get('mean') if isinstance(v, dict) else v

    rows.append({
        'cell': cell, 'combo': combo, 'cl': cl, 'ul': ul,
        'nseed': d.get('n_seeds'),
        'AA': g(clm, 'AA'), 'AA_final': g(clm, 'AA_final'),
        'BWT': g(clm, 'BWT'), 'KL': g(clm, 'KL'),
        'f_acc': g(ulm, 'forget_acc'), 'r_acc': g(ulm, 'retain_acc'),
        'mia': g(ulm, 'mia_score'),
        'commMB': g(d.get('total_comm_MB', {}) if isinstance(d.get('total_comm_MB'), dict) else {'mean': d.get('total_comm_MB')}, 'mean')
                  if d.get('total_comm_MB') is not None else None,
    })

order = {'c10_1p':0,'c10_2p':1,'c10_4p':2,'c100_1p':3,'c100_2p':4,'c100_4p':5,
         'tin_1p':6,'tin_2p':7,'tin_4p':8}
rows.sort(key=lambda r: (order.get(r['cell'], 99), r['combo']))

def fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -  "

cur = None
hdr = f"{'combo':28} {'sd':>2} {'AA':>6} {'AAfin':>6} {'BWT':>7} {'KL':>6} {'fAcc':>6} {'rAcc':>6} {'MIA':>6} {'commMB':>8}"
for r in rows:
    if r['cell'] != cur:
        cur = r['cell']
        print(f"\n#### CELL {cur}")
        print(hdr)
    print(f"{r['combo'][:28]:28} {str(r['nseed'] or '-'):>2} "
          f"{fmt(r['AA']):>6} {fmt(r['AA_final']):>6} {fmt(r['BWT']):>7} {fmt(r['KL'],2):>6} "
          f"{fmt(r['f_acc']):>6} {fmt(r['r_acc']):>6} {fmt(r['mia']):>6} "
          f"{fmt(r['commMB'],0):>8}")

print(f"\nTOTAL cells with aggregated.json: {len(rows)}")
# coverage matrix
cells = sorted(set(r['cell'] for r in rows), key=lambda c: order.get(c,99))
combos = sorted(set(r['combo'] for r in rows))
print("\n#### COVERAGE (combo x cell)")
print(f"{'combo':28} " + " ".join(f"{c:>8}" for c in cells))
present = {(r['combo'], r['cell']) for r in rows}
for cb in combos:
    print(f"{cb[:28]:28} " + " ".join(f"{'  Y':>8}" if (cb,c) in present else f"{'  .':>8}" for c in cells))
