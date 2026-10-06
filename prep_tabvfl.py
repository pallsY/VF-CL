"""Cross-organization tabular VFL benchmark (the REAL concentrated-ownership
anchor the eval reviewer demanded; synthetic prep_synth.py is the controlled axis).

Unlike image-column-split (CIFAR) or multi-view-but-redundant (mfeat) data — both
of which have near-UNIFORM per-class ownership (entropy ratio ~0.98) so the minimal
owning set S*(f) ~ P and ROAR's comm-proportional-to-|S*| saving is vacuous — a
cross-org tabular split gives each party a DISJOINT feature SEMANTICS owned by a
different "organization". Different orgs are discriminative for different classes
(soil type owns certain cover types; terrain owns others), so ownership should
genuinely CONCENTRATE. We do not assume it — probe_ownership.py MEASURES the
entropy ratio and we only commit if it is < ~0.8.

Covertype (7 forest-cover classes, 54 features) -> 6 parties by org:
    P0 terrain        : Elevation, Aspect, Slope                         (3)
    P1 hydrology      : H/V distance to hydrology                        (2)
    P2 infrastructure : distance to roadways, to fire points            (2)
    P3 illumination   : Hillshade 9am / noon / 3pm                       (3)
    P4 wilderness     : 4 wilderness-area indicators                    (4)
    P5 soil           : 40 soil-type indicators                         (40)

Columns are REORDERED so each party is one contiguous column block (split_features
slices contiguous ranges). Continuous columns z-scored; binary indicators kept 0/1.
Output npz matches the mfeat/synth schema consumed by data_utils._init_vector.

Usage (on a box with internet+sklearn):
  python prep_tabvfl.py --dataset covertype --per_class 2000 --out data/covtype_vfl
  rsync data/covtype_vfl/ to cluster, then probe with probe_ownership.py.
"""
import argparse, os
import numpy as np
from sklearn.datasets import fetch_openml

OPENML_ID = {'covertype': 1596, 'har': 1478}

# Covertype party groups by ORIGINAL column index (54 cols, see header).
COVTYPE_GROUPS = [
    ('terrain',        [0, 1, 2]),
    ('hydrology',      [3, 4]),
    ('infrastructure', [5, 9]),
    ('illumination',   [6, 7, 8]),
    ('wilderness',     list(range(10, 14))),
    ('soil',           list(range(14, 54))),
]
# which original cols are continuous (z-score) vs binary indicators (leave 0/1)
COVTYPE_CONTINUOUS = set(range(10))   # first 10 are continuous; 10..53 are binary


def load_raw(dataset, cache_dir):
    cache = os.path.join(cache_dir, f'{dataset}_raw.npz') if cache_dir else None
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return np.asarray(d['X'], float), np.asarray(d['y']).astype(str), [str(f) for f in d['feat']]
    d = fetch_openml(data_id=OPENML_ID[dataset], as_frame=False, parser='liac-arff')
    X = np.asarray(d.data, float); y = np.asarray(d.target).astype(str)
    feat = [str(f) for f in d.feature_names]
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, feat=np.array(feat, dtype=object))
    return X, y, feat


def stratified_subsample(X, y, per_class, seed):
    rng = np.random.default_rng(seed)
    classes = sorted(set(y.tolist()))
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep.append(idx[:per_class])
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def build_covtype(per_class, test_frac, snr_zscore, seed):
    cache = os.environ.get('TABCACHE')
    X, y, feat = load_raw('covertype', cache)
    X, y = stratified_subsample(X, y, per_class, seed)

    # reorder columns into contiguous party blocks; z-score continuous only.
    cols, view_names, lo, ranges = [], [], 0, []
    new_continuous = []
    for name, idxs in COVTYPE_GROUPS:
        for j in idxs:
            new_continuous.append(j in COVTYPE_CONTINUOUS)
        cols.extend(idxs)
        view_names.append(name)
        ranges.append((lo, lo + len(idxs)))
        lo += len(idxs)
    Xr = X[:, cols].astype(np.float32)
    cont = np.array(new_continuous)
    mu = Xr[:, cont].mean(0); sd = Xr[:, cont].std(0) + 1e-8
    Xr[:, cont] = (Xr[:, cont] - mu) / sd

    # labels -> 0-indexed ints in sorted class order
    classes = sorted(set(y.tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y], dtype=np.int64)

    rng = np.random.default_rng(seed)
    n = len(yi); perm = rng.permutation(n); n_te = int(n * test_frac)
    te_idx = np.sort(perm[:n_te]); tr_idx = np.sort(perm[n_te:])

    return dict(
        X=Xr, y=yi, train_idx=tr_idx, test_idx=te_idx,
        view_names=np.array(view_names),
        range_lo=np.array([r[0] for r in ranges]),
        range_hi=np.array([r[1] for r in ranges]),
        meta=np.array([f'dataset=covertype', f'n={n}', f'P={len(view_names)}',
                       f'C={len(classes)}', f'per_class={per_class}', f'seed={seed}']),
    )


# HAR semantic signal-family groups by the standard UCI ordering (0-indexed).
# Time-domain triaxial signals occupy 40-feature blocks; magnitude signals 13;
# all frequency-domain + angle features are lumped. Isolating GRAVITY (orientation)
# as its own party is the point: it owns the static postures (sit/stand/lay),
# while body-acc/jerk own the dynamic activities -> cross-class owner variation.
HAR_GROUPS = [
    ('gravity',   list(range(40, 80)) + list(range(213, 226))),   # tGravityAcc + Mag
    ('body_acc',  list(range(0, 40)) + list(range(200, 213))),    # tBodyAcc + Mag
    ('acc_jerk',  list(range(80, 120)) + list(range(226, 239))),  # tBodyAccJerk + Mag
    ('gyro',      list(range(120, 160)) + list(range(239, 252))), # tBodyGyro + Mag
    ('gyro_jerk', list(range(160, 200)) + list(range(252, 265))), # tBodyGyroJerk + Mag
    ('freq_angle', list(range(265, 561))),                        # all FFT + angle
]


def build_har_semantic(per_class, test_frac, seed):
    cache = os.environ.get('TABCACHE')
    X, y, feat = load_raw('har', cache)
    X, y = stratified_subsample(X, y, per_class, seed)
    cols, view_names, lo, ranges = [], [], 0, []
    for name, idxs in HAR_GROUPS:
        cols.extend(idxs); view_names.append(name)
        ranges.append((lo, lo + len(idxs))); lo += len(idxs)
    Xr = X[:, cols].astype(np.float32)
    Xr = (Xr - Xr.mean(0, keepdims=True)) / (Xr.std(0, keepdims=True) + 1e-8)
    classes = sorted(set(y.tolist())); cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y], dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(yi); perm = rng.permutation(n); n_te = int(n * test_frac)
    te_idx = np.sort(perm[:n_te]); tr_idx = np.sort(perm[n_te:])
    return dict(X=Xr, y=yi, train_idx=tr_idx, test_idx=te_idx,
                view_names=np.array(view_names),
                range_lo=np.array([r[0] for r in ranges]),
                range_hi=np.array([r[1] for r in ranges]),
                meta=np.array([f'dataset=har_sem', f'n={n}', f'P={len(HAR_GROUPS)}',
                               f'C={len(classes)}', f'seed={seed}']))


def build_har(per_class, test_frac, p_blocks, seed):
    """UCI HAR (6 activities, 561 sensor features). The openml copy anonymizes
    feature names (V1..V561) but preserves the standard ordering, in which each
    sensor signal's statistics are CONTIGUOUS. We split into p_blocks contiguous
    parties (~sensor families). HAR is a good cross-class-VARIATION test: static
    postures (sit/stand/lay) are discriminated by gravity-orientation signals
    while dynamic activities (walk/up/down) are discriminated by body-acceleration
    /jerk signals -> different activities should be owned by different parties
    (unlike Covertype where terrain dominates most classes)."""
    cache = os.environ.get('TABCACHE')
    X, y, feat = load_raw('har', cache)
    X, y = stratified_subsample(X, y, per_class, seed)
    D = X.shape[1]
    # contiguous near-equal blocks
    base, rem = divmod(D, p_blocks)
    ranges, view_names, lo = [], [], 0
    for p in range(p_blocks):
        w = base + (1 if p < rem else 0)
        ranges.append((lo, lo + w)); view_names.append(f'sensorgrp_{p}'); lo += w
    Xr = X.astype(np.float32)
    Xr = (Xr - Xr.mean(0, keepdims=True)) / (Xr.std(0, keepdims=True) + 1e-8)
    classes = sorted(set(y.tolist())); cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y], dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(yi); perm = rng.permutation(n); n_te = int(n * test_frac)
    te_idx = np.sort(perm[:n_te]); tr_idx = np.sort(perm[n_te:])
    return dict(X=Xr, y=yi, train_idx=tr_idx, test_idx=te_idx,
                view_names=np.array(view_names),
                range_lo=np.array([r[0] for r in ranges]),
                range_hi=np.array([r[1] for r in ranges]),
                meta=np.array([f'dataset=har', f'n={n}', f'P={p_blocks}',
                               f'C={len(classes)}', f'per_class={per_class}', f'seed={seed}']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='covertype', choices=['covertype', 'har', 'har_sem'])
    ap.add_argument('--p_blocks', type=int, default=7,
                    help='number of contiguous sensor-group parties for HAR (ignored for har_sem)')
    ap.add_argument('--per_class', type=int, default=2000)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/covtype_vfl')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.dataset == 'har_sem':
        ds = build_har_semantic(args.per_class, args.test_frac, args.seed)
    elif args.dataset == 'har':
        ds = build_har(args.per_class, args.test_frac, args.p_blocks, args.seed)
    else:
        ds = build_covtype(args.per_class, args.test_frac, None, args.seed)
    path = os.path.join(args.out, f'{args.dataset}_vfl.npz')
    np.savez_compressed(path, **ds)
    P = len(ds['view_names'])
    widths = (ds['range_hi'] - ds['range_lo']).tolist()
    print(f"wrote {path}  X={ds['X'].shape}  P={P} parties  widths={widths}")
    print(f"  parties: {list(ds['view_names'])}")
    print(f"  classes: {len(set(ds['y'].tolist()))}  train={len(ds['train_idx'])} test={len(ds['test_idx'])}")
    print(f"  rsync data/{os.path.basename(args.out)}/ to cluster; "
          f"run main.py --data tabvfl --vector_npz {path} --num_parties {P}")


if __name__ == '__main__':
    main()
