"""Joint-training ownership-entropy PROBE — the benchmark SELECTION tool.

The eval reviewer's rule: do NOT pick a VFL benchmark and HOPE ownership
concentrates (we lost on CIFAR and mfeat that way). MEASURE the per-class
ownership entropy on a plainly-trained model FIRST, and only commit if it is
genuinely concentrated (entropy ratio < ~0.8, |S*| <= P/2 for most classes).

This trains ONE joint VFL model on ALL classes (no continual-learning, no
unlearning — so the measurement is free of CL/UL dynamics; cf. the finding that
proto_evolve's feat-KD flattens late-class ownership), then runs ROAR's exact
per-party decomposition to get pi_{f,k} ∝ E_{D_f}|s_{f,k}| for every class.

Reports, per class: entropy ratio of the ownership shares, and |S*(tau)| for a
grid of tau. Writes ownership_probe.json.

Usage (uses the same CLI as main.py via get_config):
  python probe_ownership.py --data tabvfl \
    --vector_npz data/covtype_vfl/covertype_vfl.npz \
    --num_parties 6 --num_classes 7 --model_type mlp --aggregation concat \
    --cosine_head --epochs_per_task 40 --seeds 42 --device cuda:0
"""
import json, os
import numpy as np
import torch
from config import get_config
from data_utils import VFLDataset
from models import build_models
from vfl_trainer import VFLTrainer
from ul_methods.roar import RoarUL


def entropy_ratio(shares):
    p = np.asarray(shares, float); p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def s_star_size(shares, tau):
    order = np.argsort(-np.asarray(shares)); cum = 0.0
    for i, k in enumerate(order, 1):
        cum += shares[k]
        if cum >= tau:
            return i
    return len(shares)


def main():
    args = get_config()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.dataset_ref = dataset

    classes = list(range(args.num_classes))
    top.expand_classes(args.num_classes, args.device)
    train_loader, _ = dataset.get_task_loaders(classes)
    print(f"[probe] joint training {args.num_classes} classes, "
          f"P={args.num_parties}, {args.epochs_per_task} epochs ...")
    trainer.train_task(train_loader, args.epochs_per_task)

    # measure ownership on the (held-out) test split to avoid train-fit optimism
    forget_loader, _ = dataset.get_task_loaders(classes, shuffle_train=False)
    roar = RoarUL(trainer, args)
    own = roar._ownership(classes, forget_loader)

    taus = [0.5, 0.6, 0.7, 0.8, 0.9]
    P = args.num_parties
    per_class, ent_list, sstar_at = {}, [], {t: [] for t in taus}
    print(f"\n{'cls':>3} {'entropy':>8} " + ' '.join(f'|S*|@{t}' for t in taus) + '   shares')
    for c in classes:
        sh = own[c]['shares']
        er = entropy_ratio(sh); ent_list.append(er)
        sizes = {t: s_star_size(sh, t) for t in taus}
        for t in taus:
            sstar_at[t].append(sizes[t])
        per_class[c] = {'entropy_ratio': er, 'shares': sh,
                        's_star_size': sizes,
                        'top_party': int(np.argmax(sh)), 'max_share': float(np.max(sh))}
        print(f"{c:>3} {er:>8.3f} " + ' '.join(f'{sizes[t]:>6}' for t in taus) +
              '   ' + str([round(x, 3) for x in sh]))

    mean_ent = float(np.mean(ent_list))
    summary = {
        'dataset': args.data, 'P': P, 'C': args.num_classes,
        'mean_entropy_ratio': mean_ent,
        'frac_classes_concentrated_lt0.8': float(np.mean([e < 0.8 for e in ent_list])),
        'mean_Sstar_over_P': {str(t): float(np.mean(sstar_at[t]) / P) for t in taus},
        'per_class': per_class,
        'GATE_pass': bool(mean_ent < 0.8),
    }
    out = os.path.join(args.output_dir, 'ownership_probe.json')
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[probe] mean entropy ratio = {mean_ent:.3f}  "
          f"(GATE {'PASS' if mean_ent < 0.8 else 'FAIL'}: want < 0.8)")
    print(f"[probe] mean |S*|/P @tau: "
          + ', '.join(f'{t}:{np.mean(sstar_at[t])/P:.2f}' for t in taus))
    print(f"[probe] wrote {out}")


if __name__ == '__main__':
    main()
