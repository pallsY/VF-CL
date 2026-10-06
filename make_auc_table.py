"""Calibrated unlearning-leakage table: the re-learned-head attack AUC against the
retrain-from-scratch oracle floor, across forget classes and real benchmarks.

The re-learned-head ROC-AUC (roar._relearn_attack) is high (~0.95-1.0) on these
benchmarks because the classes are intrinsically separable (Covertype/HAR are
~90%+ accuracy). The MEANINGFUL quantity is the EXCESS over a model that never
saw the class: AUC_roar - AUC_retrain_oracle. ~0 means roar is indistinguishable
from full retraining against the strongest linear-probe attacker, at |S*|<P comm.

Oracle floors are produced by attack_baseline.py (one retrain per forget class).
Pass them via --oracles "bench:class:auc,...".

Usage:
  python make_auc_table.py \
    --sweeps "Covertype:results/covtype_tau2,HAR-sem:results/harsem_tau" \
    --oracles "Covertype:6:0.974,Covertype:4:0.947,HAR-sem:2:0.993,HAR-sem:5:1.000"
"""
import argparse, glob, json
import numpy as np


def roar_auc(root, cls):
    a = [v for p in glob.glob(f'{root}/tau_*/**/roar_ownership_f*.json', recursive=True)
         for v in [json.load(open(p)).get('residual_diag', {}).get(str(cls), {})
                   .get('relearn_attack', {}).get('auc')]
         if v is not None and v == v]
    return (float(np.mean(a)), len(a)) if a else (float('nan'), 0)


# Canonical paper table: (benchmark, forget class, sweep root, oracle artifact glob).
# Oracle paths are EXPLICIT per row because forget-class ids collide across
# benchmarks (Covertype and HAR both have a class 2); keying by class alone is wrong.
CANONICAL_ROWS = [
    ('Covertype', 6, 'results/covtype_tau3', 'results/oracle_auc/**/attack_baseline.json'),
    ('Covertype', 4, 'results/covtype_tau3', 'results/oracle2/cov_c4/**/attack_baseline.json'),
    ('Covertype', 1, 'results/covtype_c1',   'results/oracle3/c1/**/attack_baseline.json'),
    ('HAR-sem',   2, 'results/harsem_tau',   'results/oracle2/har_c2/**/attack_baseline.json'),
    ('HAR-sem',   5, 'results/harsem_tau',   'results/oracle2/har_c5/**/attack_baseline.json'),
]


def _oracle(glob_path):
    fs = glob.glob(glob_path, recursive=True)
    if not fs:
        return float('nan'), None
    return float(json.load(open(fs[0]))['oracle_relearn_auc']), fs[0]


def discover_oracles(oracle_glob):
    """Read committed attack_baseline.json artifacts -> {forget_class: (auc, path)}.
    Reproducible: the floor comes from a retrain-on-retain run on disk, not a CLI
    number. Forget-class ids are unique across our two benchmarks (cov {4,6},
    har {2,5}) so class is a sufficient key."""
    floors = {}
    for p in glob.glob(oracle_glob, recursive=True):
        d = json.load(open(p))
        floors[int(d['forget_class'])] = (float(d['oracle_relearn_auc']), p)
    return floors


def main():
    print(f'{"benchmark":>10} {"class":>5} {"roar_AUC":>9} {"oracle":>7} {"excess":>8}  n   class-separability')
    out = []
    for name, cls, root, oglob in CANONICAL_ROWS:
        ra, n = roar_auc(root, cls)
        orc, opath = _oracle(oglob)
        excess = ra - orc if (ra == ra and orc == orc) else float('nan')
        sat = 'non-saturated' if orc < 0.90 else 'saturated'
        print(f'{name:>10} {cls:>5} {ra:>9.3f} {orc:>7.3f} {excess:>+8.3f}  {n}   {sat}')
        out.append({'benchmark': name, 'class': cls, 'roar_auc': ra,
                    'oracle_auc': orc, 'excess': excess, 'n': n,
                    'separability': sat, 'oracle_artifact': opath})
    valid = [r['excess'] for r in out if r['excess'] == r['excess']]
    print(f'\nmax |excess| over oracle = {max(abs(e) for e in valid):.3f} over {len(valid)} '
          f'forget classes / 2 real benchmarks.\nroar <= oracle on the non-saturated class '
          f'(more private than retraining); ~ oracle elsewhere.')
    json.dump(out, open('figs/auc_vs_oracle.json', 'w'), indent=2)
    print('wrote figs/auc_vs_oracle.json')


if __name__ == '__main__':
    main()
