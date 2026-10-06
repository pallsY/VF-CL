"""Post-hoc KL injection for the fragmentation sweep.

The fragmentation driver runs Oracle and each method as separate single-combo
invocations, which don't trigger main.py's `inject_kl_to_oracle`. This walks
every results/frag/<dsname>_<P>p/ cell, finds the Oracle final_probs.npz for
that fragmentation level, computes KL(p_oracle || p_method) per seed, averages,
and writes a `KL` field into each method's aggregated.json.

Idempotent: safe to re-run after each new cell completes.
"""
import os, sys, glob, json
import numpy as np


def compute_kl_to_oracle(oracle_probs, method_probs, retained_classes, eps=1e-8):
    """Mean per-sample KL(p_oracle || p_method) on the retained-class support.
    Pure-numpy copy of metrics.compute_kl_to_oracle so this script needs no torch
    (the DCU torch build won't import without the DTK env)."""
    if oracle_probs.shape[0] == 0 or method_probs.shape[0] == 0:
        return None
    n = min(oracle_probs.shape[0], method_probs.shape[0])
    cols = list(retained_classes)
    po = oracle_probs[:n][:, cols].astype(np.float64)
    pm = method_probs[:n][:, cols].astype(np.float64)
    po = po / (po.sum(1, keepdims=True) + eps)
    pm = pm / (pm.sum(1, keepdims=True) + eps)
    po = np.clip(po, eps, 1.0); pm = np.clip(pm, eps, 1.0)
    kl = (po * (np.log(po) - np.log(pm))).sum(1)
    return float(np.mean(kl))


def find_final_probs(combo_dir):
    """Return list of (seed, npz_path) for this combo's per-seed final_probs."""
    out = []
    for p in glob.glob(os.path.join(combo_dir, '*', 'seed_*', 'final_probs.npz')) + \
             glob.glob(os.path.join(combo_dir, '*', '*', 'seed_*', 'final_probs.npz')):
        seed = int(os.path.basename(os.path.dirname(p)).split('_')[1])
        out.append((seed, p))
    return out


def find_aggregated(combo_dir):
    """Return the aggregated.json path (or None)."""
    cands = (glob.glob(os.path.join(combo_dir, '*', 'aggregated.json')) +
             glob.glob(os.path.join(combo_dir, '*', '*', 'aggregated.json')))
    return cands[0] if cands else None


def process_level(level_dir):
    """level_dir = .../results/frag/<dsname>_<P>p/"""
    name = os.path.basename(level_dir.rstrip('/'))
    oracle_probs = {}
    for seed, p in find_final_probs(os.path.join(level_dir, 'oracle')):
        oracle_probs[seed] = np.load(p)
    if not oracle_probs:
        print(f"  [{name}] no oracle yet, skip"); return

    for combo in sorted(os.listdir(level_dir)):
        if combo == 'oracle': continue
        cdir = os.path.join(level_dir, combo)
        agg_path = find_aggregated(cdir)
        if not agg_path: continue
        kl_vals = []
        for seed, p in find_final_probs(cdir):
            if seed not in oracle_probs: continue
            od = oracle_probs[seed]; md = np.load(p)
            retained = od['retained'].tolist()
            kl = compute_kl_to_oracle(od['probs'], md['probs'], retained)
            if kl is not None: kl_vals.append(kl)
        if not kl_vals: continue
        agg = json.load(open(agg_path))
        agg.setdefault('cl_metrics', {})['KL'] = {
            'mean': round(float(np.mean(kl_vals)), 4),
            'std':  round(float(np.std(kl_vals)),  4),
            'values': [round(v, 4) for v in kl_vals],
        }
        json.dump(agg, open(agg_path, 'w'), indent=2, default=str)
        print(f"  [{name}] KL injected for {combo}: {agg['cl_metrics']['KL']['mean']:.3f}")


if __name__ == '__main__':
    # base dir: CLI arg, else $FRAG_DIR, else ./results/frag relative to this script
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        base = os.environ.get('FRAG_DIR') or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results', 'frag')
    if not os.path.isdir(base):
        sys.exit(f"frag dir not found: {base}")
    print(f"KL injection over {base}")
    for level in sorted(os.listdir(base)):
        lvl = os.path.join(base, level)
        if os.path.isdir(lvl):
            process_level(lvl)
