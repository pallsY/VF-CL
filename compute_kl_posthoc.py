"""Post-hoc KL-to-oracle over the benchmark result trees (run ON the cluster).

For every (suite, dataset, order, cl, ul, seed) run with a saved
final_probs.npz, computes mean KL(p_oracle || p_method) on the retained-class
support, where the oracle reference is the SAME row's `retrain` run with the
SAME seed (same queue -> same retained set, shuffle=False -> row-aligned).
Emits kl_posthoc.json: {"<combo_dir>": {"<seed>": kl, ...}, ...}
"""
import glob, json, os
import numpy as np


def best_ms(combo_dir):
    cands = []
    for ms in sorted(glob.glob(os.path.join(combo_dir, '*_multiseed_*'))):
        seeds = glob.glob(os.path.join(ms, '*', 'seed_*', 'final_probs.npz'))
        if seeds:
            cands.append((len(seeds), ms, seeds))
    if not cands:
        return {}
    n = max(c[0] for c in cands)
    _, _, seeds = [c for c in cands if c[0] == n][-1]
    return {os.path.basename(os.path.dirname(f)): f for f in seeds}  # seed_42 -> path


def kl(po, pm, cols, eps=1e-8):
    n = min(po.shape[0], pm.shape[0])
    po = po[:n][:, cols].astype(np.float64); pm = pm[:n][:, cols].astype(np.float64)
    po /= (po.sum(1, keepdims=True) + eps); pm /= (pm.sum(1, keepdims=True) + eps)
    po = np.clip(po, eps, 1.0); pm = np.clip(pm, eps, 1.0)
    return float(np.mean((po * (np.log(po) - np.log(pm))).sum(1)))


def row_groups():
    """Yield (oracle_combo_dir, [method_combo_dirs]) sharing one queue+backbone."""
    for dso in glob.glob('results/grid_tab/*_o[12]') + glob.glob('results/grid_cifar/o[12]'):
        combos = glob.glob(os.path.join(dso, '*__*'))
        by_cl = {}
        for c in combos:
            cl = os.path.basename(c).split('__')[0]
            by_cl.setdefault(cl, []).append(c)
        for cl, cs in by_cl.items():
            oracle = os.path.join(dso, f'{cl}__retrain')
            if os.path.isdir(oracle):
                yield oracle, cs
    for ds in glob.glob('results/appendix/ulonly_*'):
        oracle = os.path.join(ds, 'retrain')
        if os.path.isdir(oracle):
            yield oracle, glob.glob(os.path.join(ds, '*'))


def main():
    out = {}
    for oracle_dir, method_dirs in row_groups():
        oref = best_ms(oracle_dir)
        if not oref:
            continue
        ocache = {}
        for md in method_dirs:
            mref = best_ms(md)
            vals = {}
            for seed, mp in mref.items():
                if seed not in oref:
                    continue
                if seed not in ocache:
                    d = np.load(oref[seed])
                    ocache[seed] = (d['probs'], d['retained'].tolist())
                po, cols = ocache[seed]
                dm = np.load(mp)
                try:
                    vals[seed.replace('seed_', '')] = round(kl(po, dm['probs'], cols), 4)
                except Exception as e:
                    print('skip', mp, e)
            if vals:
                out[md] = vals
    with open('kl_posthoc.json', 'w') as f:
        json.dump(out, f, indent=0)
    print('combos with KL:', len(out))


if __name__ == '__main__':
    main()
