"""Cross-seed rank-stability check for the per-class redundancy profile.

The single-seed analyzer (analyze_redundancy.py) showed Spearman rho ~0.9
between cells (P=2 vs P=4). The "intrinsic" claim, however, hinges on the
profile being stable across SEEDS too — otherwise the same class ranks at
position 7 in seed 42 and position 80 in seed 43, and there's no per-class
structure to exploit.

This script reads 3+ probe_report.json files (different seeds, same cell)
and reports:
  - pairwise Spearman rank corr between (seed_i, seed_j) on per-class
    max-party-acc;
  - mean +/- std of those pairwise correlations;
  - "consistently easy" classes (top-quartile by max-party-acc in ALL seeds);
  - "consistently hard" classes (bottom-quartile in ALL seeds);
  - per-class max-party-acc table (across seeds) with mean and CV.

Usage:
    python analyze_seed_stability.py \
        ./results/probe_c100_p4_42/probe_report.json \
        ./results/probe_c100_p4_s43_XXX/probe_report.json \
        ./results/probe_c100_p4_s44_XXX/probe_report.json
"""
import json, sys, os
import numpy as np
from itertools import combinations


def load(p):
    with open(p) as f:
        r = json.load(f)
    cfg = r['config']
    pcm = {int(c): v for c, v in r['q2']['per_class_max_party_acc'].items()}
    return {
        'path': p,
        'seed': cfg.get('seed'),
        'data': cfg.get('data'),
        'P': cfg.get('num_parties'),
        'per_class_max': pcm,
    }


def spearman_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra*ra).sum()) * np.sqrt((rb*rb).sum())
    return float((ra*rb).sum() / denom) if denom > 0 else 0.0


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("Need >= 2 probe_report.json paths for cross-seed analysis.")
        sys.exit(1)

    reports = [load(p) for p in paths]
    print(f"Loaded {len(reports)} reports:")
    for r in reports:
        print(f"  seed={r['seed']} cell={r['data']}_P{r['P']}  {r['path']}")

    # Build a unified per-class matrix: classes x seeds
    classes = sorted(set().union(*[set(r['per_class_max'].keys()) for r in reports]))
    seeds = [r['seed'] for r in reports]
    mat = np.full((len(classes), len(reports)), np.nan)
    for ci, c in enumerate(classes):
        for si, r in enumerate(reports):
            v = r['per_class_max'].get(c)
            if v is not None:
                mat[ci, si] = v

    # Pairwise Spearman rank correlations
    print("\n=== Pairwise Spearman rank correlations (per-class max-party-acc) ===")
    rhos = []
    for (i, ri), (j, rj) in combinations(enumerate(reports), 2):
        col_i = mat[:, i]; col_j = mat[:, j]
        good = ~(np.isnan(col_i) | np.isnan(col_j))
        rho = spearman_rho(col_i[good], col_j[good])
        rhos.append(rho)
        print(f"  seed {ri['seed']:>4} <-> seed {rj['seed']:>4}:  rho = {rho:+.3f}  "
              f"(n={int(good.sum())})")
    rhos = np.array(rhos)
    print(f"\n  mean pairwise rho = {rhos.mean():+.3f}")
    print(f"  std  pairwise rho = {rhos.std():.3f}")

    verdict = (
        "STRONG  - per-class profile is genuinely intrinsic"   if rhos.mean() > 0.7 else
        "MODERATE - some seed-stability; method should still work" if rhos.mean() > 0.4 else
        "WEAK    - was likely a single-seed fluke; reconsider"
    )
    print(f"\n  VERDICT: {verdict}")

    # Per-class summary
    means = np.nanmean(mat, axis=1)
    stds = np.nanstd(mat, axis=1)
    cv = stds / np.where(means > 1e-9, means, 1.0)

    # Consistently-easy / hard classes
    # Easy: in TOP quartile in EVERY seed
    # Hard: in BOTTOM quartile in EVERY seed
    consistently_easy = []
    consistently_hard = []
    for ci, c in enumerate(classes):
        row = mat[ci, :]
        if np.isnan(row).any():
            continue
        # per-column quartile positions of this class
        in_top = []
        in_bot = []
        for j in range(mat.shape[1]):
            col = mat[:, j]
            col_no_nan = col[~np.isnan(col)]
            q75 = np.quantile(col_no_nan, 0.75)
            q25 = np.quantile(col_no_nan, 0.25)
            in_top.append(col[ci] >= q75)
            in_bot.append(col[ci] <= q25)
        if all(in_top): consistently_easy.append((c, row.mean()))
        if all(in_bot): consistently_hard.append((c, row.mean()))

    print(f"\n=== Consistently EASY classes (top quartile in all {len(seeds)} seeds) ===")
    for c, mu in sorted(consistently_easy, key=lambda x: -x[1]):
        print(f"  class {c:>3}:  mean={mu:.3f}  vals=[{','.join(f'{v:.2f}' for v in mat[classes.index(c)])}]")
    if not consistently_easy:
        print("  (none)")

    print(f"\n=== Consistently HARD classes (bottom quartile in all {len(seeds)} seeds) ===")
    for c, mu in sorted(consistently_hard, key=lambda x: x[1]):
        print(f"  class {c:>3}:  mean={mu:.3f}  vals=[{','.join(f'{v:.2f}' for v in mat[classes.index(c)])}]")
    if not consistently_hard:
        print("  (none)")

    # Distribution
    print(f"\n=== Aggregate stats ===")
    print(f"  classes total: {len(classes)}")
    print(f"  consistently easy : {len(consistently_easy)}  ({100*len(consistently_easy)/len(classes):.1f}%)")
    print(f"  consistently hard : {len(consistently_hard)}  ({100*len(consistently_hard)/len(classes):.1f}%)")
    print(f"  per-class CV of max-party-acc across seeds: "
          f"mean={np.nanmean(cv):.3f}, max={np.nanmax(cv):.3f}")
    print(f"    (low CV -> redundancy profile is robust to random init)")


if __name__ == '__main__':
    main()
