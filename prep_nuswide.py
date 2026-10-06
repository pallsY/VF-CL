"""NUS-WIDE multi-modal VFL benchmark (the standard-VFL credibility anchor).

NUS-WIDE (Chua et al., CIVR'09): 269,648 Flickr images, 81 concepts, officially
distributed pre-extracted features. The VFL literature's canonical
feature-partitioned dataset: image low-level features vs. 1000-d tag bag-of-words
live on different parties. We extend the classic 2-party split to 6 parties:

    P0 color_hist   : 64-D color histogram            (Normalized_CH.dat)
    P1 color_corr   : 144-D color auto-correlogram    (Normalized_CORR.dat)
    P2 edge_hist    : 73-D edge direction histogram   (Normalized_EDH.dat)
    P3 wavelet      : 128-D wavelet texture           (Normalized_WT.dat)
    P4 color_mom    : 225-D block-wise color moments  (Normalized_CM55.dat)
    P5 tags         : 1000-D tag bag-of-words         (AllTags1k.txt)

All source files are in ALL-IMAGE order (row-aligned with Imagelist.txt);
official train/test membership is recovered by matching TrainImagelist.txt
paths. (The archive's Normalized_*.dat are NOT pre-split into Train/Test —
only the BoW files are.)

Class subset: NUS-WIDE is multi-label; following the FTL/FedCVT convention we
keep images with EXACTLY ONE positive among a fixed 10-concept set, giving a
single-label 10-class task. The concept set below was chosen by exhaustive
search over 10-subsets of the top-16 most frequent concepts, maximizing the
minimum per-class count: min class = 3,900 exactly-one images (the naive
top-10-by-frequency set bottoms out at 287).

Sources: official host is dead; every archive is served by Internet Archive
snapshots of dl.nextcenter.org (Groundtruth.zip, NUS_WID_Tags.zip,
ImageList.zip, ConceptsList.zip, NUS_WID_Low_Level_Features.rar ~1.0GB).

Usage:
  python prep_nuswide.py --raw_dir <dir with Groundtruth/ NUS_WID_Tags/ ImageList/ Low_Level_Features/> \
      --per_class 3000 --per_class_test 800 --out data/nuswide_vfl
  rsync data/nuswide_vfl/ to cluster; run main.py --data tabvfl \
      --vector_npz data/nuswide_vfl/nuswide_vfl.npz --num_parties 6 --model_type mlp
"""
import argparse, os
import numpy as np

# chosen by exhaustive search (see module docstring)
CONCEPTS10 = ['person', 'animal', 'buildings', 'window', 'ocean',
              'road', 'flowers', 'sunset', 'reflection', 'rocks']

IMAGE_VIEWS = [
    ('color_hist', 'Normalized_CH.dat',   64),
    ('color_corr', 'Normalized_CORR.dat', 144),
    ('edge_hist',  'Normalized_EDH.dat',  73),
    ('wavelet',    'Normalized_WT.dat',   128),
    ('color_mom',  'Normalized_CM55.dat', 225),
]
N_ALL = 269648


def _load_matrix(path, expect_dim, dtype=np.float32):
    """Robust text-matrix load; tolerates a stray extra column (some official
    files carry a trailing/index column -> drop the offender, assert the rest)."""
    m = np.loadtxt(path, dtype=dtype)
    assert m.shape[0] == N_ALL, f'{path}: {m.shape[0]} rows, expected {N_ALL}'
    if m.shape[1] == expect_dim + 1:
        # drop whichever edge column is constant/degenerate; else drop the last
        if np.all(m[:, -1] == m[0, -1]):
            m = m[:, :-1]
        elif np.all(m[:, 0] == m[0, 0]):
            m = m[:, 1:]
        else:
            m = m[:, :expect_dim]
    assert m.shape[1] == expect_dim, f'{path}: {m.shape[1]} cols, expected {expect_dim}'
    return m


def _norm_key(line):
    """Imagelist path -> comparable key (strip drive prefix, unify separators)."""
    return line.strip().replace('\\', '/').split('Flickr/')[-1].lower()


def load_all(raw_dir):
    gd = os.path.join(raw_dir, 'Groundtruth', 'AllLabels')
    lab = np.stack([np.loadtxt(os.path.join(gd, f'Labels_{c}.txt'), dtype=np.int8)
                    for c in CONCEPTS10], 1)                      # N_ALL x 10
    assert lab.shape[0] == N_ALL

    il = os.path.join(raw_dir, 'ImageList')
    with open(os.path.join(il, 'Imagelist.txt'), encoding='latin-1') as f:
        all_keys = [_norm_key(l) for l in f]
    with open(os.path.join(il, 'TrainImagelist.txt'), encoding='latin-1') as f:
        train_keys = {_norm_key(l) for l in f}
    train_mask = np.array([k in train_keys for k in all_keys])
    assert train_mask.sum() == 161789, f'train mask {train_mask.sum()} != 161789'
    return lab, train_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--per_class', type=int, default=3000)
    ap.add_argument('--per_class_test', type=int, default=800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/nuswide_vfl')
    args = ap.parse_args()

    lab, train_mask = load_all(args.raw_dir)
    sel = lab.sum(1) == 1
    y_all = lab.argmax(1).astype(np.int64)
    print(f'exactly-one rows: {sel.sum()} of {N_ALL}')

    # class-balanced subsample within official train / test streams
    rng = np.random.default_rng(args.seed)
    rows_tr, rows_te = [], []
    for c in range(len(CONCEPTS10)):
        idx = np.where(sel & (y_all == c) & train_mask)[0]
        rng.shuffle(idx); rows_tr.append(idx[:args.per_class])
        idx = np.where(sel & (y_all == c) & ~train_mask)[0]
        rng.shuffle(idx); rows_te.append(idx[:args.per_class_test])
    rows_tr = np.concatenate(rows_tr); rng.shuffle(rows_tr)
    rows_te = np.concatenate(rows_te); rng.shuffle(rows_te)
    rows = np.concatenate([rows_tr, rows_te])

    # features: 5 image views + tags, all in all-image order
    blocks = []
    for name, fname, dim in IMAGE_VIEWS:
        p = os.path.join(args.raw_dir, 'Low_Level_Features', fname)
        print(f'loading {fname} ...')
        blocks.append(_load_matrix(p, dim)[rows])
    print('loading AllTags1k.txt ...')
    tags = _load_matrix(os.path.join(args.raw_dir, 'NUS_WID_Tags', 'AllTags1k.txt'),
                        1000)[rows]

    n_tr = len(rows_tr)
    Xi = np.concatenate(blocks, 1)
    # z-score image features on TRAIN stats; tags stay binary 0/1
    mu, sd = Xi[:n_tr].mean(0), Xi[:n_tr].std(0) + 1e-8
    Xi = (Xi - mu) / sd
    X = np.concatenate([Xi, tags], 1).astype(np.float32)
    y = y_all[rows]

    view_names, ranges, lo = [], [], 0
    for name, _, dim in IMAGE_VIEWS + [('tags', '', 1000)]:
        view_names.append(name); ranges.append((lo, lo + dim)); lo += dim

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'nuswide_vfl.npz')
    np.savez_compressed(
        path, X=X, y=y,
        train_idx=np.arange(n_tr), test_idx=np.arange(n_tr, len(rows)),
        view_names=np.array(view_names),
        range_lo=np.array([r[0] for r in ranges]),
        range_hi=np.array([r[1] for r in ranges]),
        meta=np.array([f'dataset=nuswide', f'n={len(rows)}', f'P={len(view_names)}',
                       f'C={len(CONCEPTS10)}', f'concepts={",".join(CONCEPTS10)}',
                       f'per_class={args.per_class}', f'seed={args.seed}']),
    )
    print(f'wrote {path}  X={X.shape}  train={n_tr} test={len(rows)-n_tr}')
    print(f'  parties: {view_names}  widths={[r[1]-r[0] for r in ranges]}')
    print(f'  train per-class: {np.bincount(y[:n_tr], minlength=10).tolist()}')
    print(f'  test  per-class: {np.bincount(y[n_tr:], minlength=10).tolist()}')


if __name__ == '__main__':
    main()
