"""Synthetic feature-heterogeneous VFL benchmark with a CONTROLLABLE ownership-
specialization knob rho.

Motivation (the experimental spine the paper was missing). On image-column-split
VFL (CIFAR) and on mfeat-6view, measured per-class ownership is near-UNIFORM
(entropy ratio ~0.98) — every party is individually informative for every class,
so the minimal owning set S*(f) ~ P and ROAR's "communication proportional to
|S*|" saving is vacuous. That is not a quirk of two datasets; it is what happens
whenever parties hold REDUNDANT views. The mechanism needs party-per-class
SPECIALIZATION, which off-the-shelf sets do not guarantee.

This generator makes specialization a tunable axis rho in [0,1]:

  - Each class c is assigned a primary owning set owners(c) (size n_owners).
  - Party k carries class-c discriminative ENERGY ||mu_{c,k}||^2 = SNR^2 * w_{c,k},
    with mixing weight  w_{c,k} = (1-rho)/P + rho * 1[k in owners(c)] / n_owners,
    so sum_k w_{c,k} = 1 for every c.  =>  total per-class energy = SNR^2 is
    CONSTANT in rho: difficulty is fixed; rho only redistributes WHERE the signal
    lives.
  - rho=0 -> uniform ownership (reproduces the CIFAR/mfeat null regime, S*=P).
  - rho=1 -> full specialization (only owners(c) separate class c, |S*|=n_owners).

Each party's features: x_k = mu_{y,k} + N(0, I_d).  X is z-scored and the views
are concatenated, matching the mfeat npz schema consumed by data_utils.VFLDataset
(_init_vector): X, y, train_idx, test_idx, view_names, range_lo, range_hi.

SNR is chosen so the model is NOT saturated (~80-90% acc) — addressing the
mfeat-saturation critique: methods can now be differentiated on accuracy too.

Usage:
  python prep_synth.py --rhos 0,0.25,0.5,0.75,1.0 --P 8 --C 10 --d 16 \
      --n_train 4000 --n_test 1000 --snr 2.2 --n_owners 1 --out data/synthvfl
"""
import argparse, os
import numpy as np


def make_dataset(rho, P, C, d, n_train, n_test, snr, n_owners, seed):
    rng = np.random.default_rng(seed)
    N = n_train + n_test

    # primary owners per class: rotate so ownership is spread across parties and
    # different classes own different parties (the structure the decomposition reads).
    owners = {c: [(c * n_owners + j) % P for j in range(n_owners)] for c in range(C)}

    # mixing weights w_{c,k}: (1-rho) uniform background + rho on the owning set.
    W = np.full((C, P), (1.0 - rho) / P)
    for c in range(C):
        for k in owners[c]:
            W[c, k] += rho / n_owners
    # W rows already sum to 1 (uniform background sums to (1-rho); owner mass = rho).

    # per (class,party) mean-shift direction (unit) and magnitude sqrt(energy).
    U = rng.standard_normal((C, P, d))
    U /= (np.linalg.norm(U, axis=2, keepdims=True) + 1e-12)
    mag = snr * np.sqrt(W)                         # (C,P): ||mu_{c,k}|| = snr*sqrt(w)
    MU = U * mag[:, :, None]                       # (C,P,d)

    # balanced labels
    y = np.repeat(np.arange(C), int(np.ceil(N / C)))[:N]
    rng.shuffle(y)
    X = np.empty((N, P * d), dtype=np.float32)
    for k in range(P):
        Xk = MU[y, k, :] + rng.standard_normal((N, d))   # mu_{y,k} + noise
        X[:, k * d:(k + 1) * d] = Xk

    # z-score per column (global), then fixed stratified train/test split.
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    idx = rng.permutation(N)
    tr_idx = np.sort(idx[:n_train]); te_idx = np.sort(idx[n_train:])

    view_names = np.array([f'party_{k}' for k in range(P)])
    range_lo = np.array([k * d for k in range(P)])
    range_hi = np.array([(k + 1) * d for k in range(P)])

    # ground-truth ownership (the generative weights) for sanity vs measured.
    return dict(X=X.astype(np.float32), y=y.astype(np.int64),
                train_idx=tr_idx, test_idx=te_idx,
                view_names=view_names, range_lo=range_lo, range_hi=range_hi,
                gt_ownership=W.astype(np.float32),
                meta=np.array([f'rho={rho}', f'P={P}', f'C={C}', f'd={d}',
                               f'snr={snr}', f'n_owners={n_owners}', f'seed={seed}']))


def entropy_ratio(w):
    p = np.asarray(w) / (np.sum(w) + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rhos', default='0,0.25,0.5,0.75,1.0')
    ap.add_argument('--P', type=int, default=8)
    ap.add_argument('--C', type=int, default=10)
    ap.add_argument('--d', type=int, default=16)
    ap.add_argument('--n_train', type=int, default=4000)
    ap.add_argument('--n_test', type=int, default=1000)
    ap.add_argument('--snr', type=float, default=2.2)
    ap.add_argument('--n_owners', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/synthvfl')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rhos = [float(r) for r in args.rhos.split(',')]

    for rho in rhos:
        ds = make_dataset(rho, args.P, args.C, args.d, args.n_train, args.n_test,
                          args.snr, args.n_owners, args.seed)
        tag = f'{rho:.2f}'.replace('.', 'p')
        path = os.path.join(args.out, f'synth_rho{tag}.npz')
        np.savez_compressed(path, **ds)
        gt_er = np.mean([entropy_ratio(ds['gt_ownership'][c]) for c in range(args.C)])
        print(f"rho={rho:.2f} -> {path}  X={ds['X'].shape}  "
              f"GT mean entropy_ratio={gt_er:.3f} "
              f"(expect ~1.0 at rho=0, ~0 at rho=1)")
    print(f"\nGenerated {len(rhos)} files in {args.out}/. "
          f"rsync to cluster: data/synthvfl/  then sweep --vector_npz per rho.")


if __name__ == '__main__':
    main()
