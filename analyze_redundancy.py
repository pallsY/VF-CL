"""Compare Q2 per-class redundancy profiles across fragmentation cells.

Reads every probe_report.json under results/probe_*/ and produces:

  1. Per-cell heterogeneity table — does the std/range of per-class
     single-party-recognizability vary with P or dataset?
  2. "Class identity persistence" — for shared classes across cells of the
     same dataset, are the "easy" classes consistently easy (intrinsic
     redundancy structure) or do they shuffle (cell-specific artifact)?

The user's question: does the heterogeneity in single-party probe accuracy
correlate with anything structural? If "yes, certain classes are robustly
single-party-recognizable across cells" -> there is an intrinsic per-class
redundancy axis we can build a method on. If "no, heterogeneity is noise"
-> the structure we saw in P=2/P=4 was a fluke.
"""
import glob, json, os, sys
import numpy as np
from collections import defaultdict


def find_reports(root='./results'):
    return sorted(glob.glob(os.path.join(root, 'probe_*', 'probe_report.json')))


def load_report(p):
    with open(p) as f:
        r = json.load(f)
    cfg = r['config']
    # accuracy matrix may have string keys after json round-trip
    pcpa = r['q2'].get('per_class_per_party_acc', {})
    pcpa = {int(c): {int(k): v for k, v in d.items()} for c, d in pcpa.items()}
    return {
        'path': p,
        'data': cfg.get('data'),
        'P': cfg.get('num_parties'),
        'agg': cfg.get('aggregation'),
        'cl': cfg.get('cl_method'),
        'seed': cfg.get('seed'),
        'q1': r['decision_summary'].get('final_avg_max_share'),
        'q1_uniform': r['decision_summary'].get('uniform_share_baseline'),
        'q1_verdict': r['decision_summary'].get('q1_verdict'),
        'q2_verdict': r['decision_summary'].get('q2_verdict'),
        'per_class_max': r['q2'].get('per_class_max_party_acc', {}),
        'per_class_per_party': pcpa,
        'het': r['q2'].get('heterogeneity', {}),
        'sanity': r['sanity'].get('mean_pairwise_party_embedding_cosine'),
    }


def print_per_cell_table(reports):
    print("=" * 110)
    print(f"{'cell':<22} {'P':>3} {'agg':>5} {'Q1':>9} {'Q1<unif>':>9} "
          f"{'mpa_mean':>9} {'mpa_std':>9} {'mpa_range':>10} {'easy(.6+)':>10} {'hard(.3-)':>10}")
    print("-" * 110)
    for r in reports:
        name = f"{r['data']}_p{r['P']}_{r['agg']}"
        h = r['het']
        if not h:
            continue
        print(f"{name:<22} {r['P']:>3} {r['agg']:>5} {r['q1']:>9.3f} {r['q1_uniform']:>9.3f} "
              f"{h.get('max_party_acc_mean', 0):>9.3f} "
              f"{h.get('max_party_acc_std', 0):>9.3f} "
              f"{h.get('max_party_acc_range', 0):>10.3f} "
              f"{len(h.get('easy_classes_above_0.60', [])):>10d} "
              f"{len(h.get('hard_classes_below_0.30', [])):>10d}")
    print("=" * 110)


def class_persistence(reports, dataset):
    """For a single dataset (different P values), check if the same classes
    are "easy" / "hard" across cells. If yes -> per-class redundancy is
    INTRINSIC, not cell-dependent."""
    subs = [r for r in reports if r['data'] == dataset]
    if len(subs) < 2:
        return
    # Build per-class score map: class -> {cell_name -> max_party_acc}
    all_classes = set()
    for r in subs:
        all_classes.update(int(c) for c in r['per_class_max'].keys())
    rows = []
    for c in sorted(all_classes):
        vals = []
        for r in subs:
            key = str(c) if str(c) in r['per_class_max'] else c
            v = r['per_class_max'].get(key)
            vals.append(v if v is not None else float('nan'))
        rows.append((c, vals))
    print(f"\n  Per-class max-party-acc, dataset={dataset}")
    cells = ' '.join(f"P{r['P']}_{r['agg']}".rjust(10) for r in subs)
    print(f"  class | {cells} | rank-corr-with-P{subs[0]['P']}")
    print(f"  {'-'*5} | {'-' * (10*len(subs) + len(subs)-1)} | ---")
    # Spearman rank correlation between (P=lowest) ordering and other cells
    base_vals = np.array([v[1][0] for v in rows])
    base_rank = np.argsort(np.argsort(base_vals))
    for c, vals in rows:
        row_str = ' '.join((f"{v:>10.3f}" if not np.isnan(v) else f"{'N/A':>10}") for v in vals)
        print(f"  {c:>5} | {row_str}")
    # Per-cell rank correlation to base cell
    print(f"\n  Spearman rank corr of per-class max-party-acc vs cell P={subs[0]['P']}:")
    for r in subs[1:]:
        v = np.array([r['per_class_max'].get(str(c), r['per_class_max'].get(c, np.nan))
                       for c in sorted(all_classes)])
        if np.isnan(v).any():
            print(f"    P={r['P']}: missing classes, skip"); continue
        rk = np.argsort(np.argsort(v))
        # Spearman = pearson on ranks
        a, b = base_rank.astype(float), rk.astype(float)
        a -= a.mean(); b -= b.mean()
        denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()))
        rho = float((a*b).sum() / denom) if denom > 0 else 0.0
        print(f"    P={r['P']}: rho = {rho:+.3f}  (1.0 = identical class ordering, 0 = unrelated)")


def party_skew(reports):
    """Per cell: does ONE party consistently dominate across many classes?
    (This is the positional bias we saw in the first two probes.)"""
    print("\n  Per-cell winning-party concentration (Q1 argmax):")
    print(f"  {'cell':<22} {'argmax_party_distribution_over_classes':<40}")
    for r in reports:
        # need argmax_party from per_task_ownership last snapshot. We can read from raw json.
        with open(r['path']) as f:
            raw = json.load(f)
        snap = raw['per_task_ownership'][-1]['per_class']
        argmax = [snap[str(c)]['argmax_party'] if str(c) in snap else snap[c]['argmax_party']
                  for c in snap]
        counts = [argmax.count(k) for k in range(r['P'])]
        print(f"  {r['data']}_p{r['P']}_{r['agg']:<5}      {counts}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else './results'
    reports = [load_report(p) for p in find_reports(root)]
    if not reports:
        print(f"No probe_report.json under {root}/probe_*/. Run probe_attribution.py first.")
        return
    print(f"\nFound {len(reports)} probe reports under {root}.\n")
    print_per_cell_table(reports)
    # Per-dataset persistence analysis
    for ds in sorted({r['data'] for r in reports}):
        class_persistence(reports, ds)
    party_skew(reports)


if __name__ == '__main__':
    main()
