"""Calibration floor for the re-learned-head attack: the RETRAIN-from-scratch
oracle's class-f recoverability AUC.

The re-learned-head attack (roar._relearn_attack) trains the best linear
forget-vs-rest probe on the post-unlearning embeddings. An AUC near 1.0 looks
like leakage — but only if a model that NEVER saw class f would score near 0.5.
If class f is intrinsically separable from the retain classes by raw features
(e.g. Covertype class6 = Krummholz, extreme-elevation), even the retrain oracle's
embeddings separate it and AUC is high REGARDLESS of unlearning, making AUC the
wrong leakage metric (the head-reconnection R_f certificate is then the right one).

This trains a model on RETAIN classes only (forget class fully excluded), then
runs the same re-learned-head probe for the forget class -> the AUC FLOOR.

Usage: python attack_baseline.py --data tabvfl --vector_npz ... --num_parties 6 \
  --num_classes 7 --model_type mlp --aggregation concat --cosine_head \
  --forget_class 6 --epochs_per_task 40 --seeds 42 --device cuda:0
"""
import json, os
import numpy as np
import torch
from config import get_config
from data_utils import VFLDataset
from models import build_models
from vfl_trainer import VFLTrainer
from ul_methods.roar import RoarUL


def main():
    args = get_config()
    fc = int(getattr(args, 'forget_class', 6))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.dataset_ref = dataset

    retain = [c for c in range(args.num_classes) if c != fc]
    top.expand_classes(args.num_classes, args.device)
    train_loader, _ = dataset.get_task_loaders(retain)   # RETAIN ONLY — oracle
    print(f"[oracle] training on retain classes {retain} (class {fc} EXCLUDED), "
          f"{args.epochs_per_task} epochs ...")
    trainer.train_task(train_loader, args.epochs_per_task)

    # attack loader: all classes' test split (incl. the never-trained forget class)
    atk_loader = dataset.get_task_loaders(list(range(args.num_classes)),
                                          shuffle_train=False)[1]
    roar = RoarUL(trainer, args)
    res = roar._relearn_attack([fc], atk_loader)
    print(f"[oracle] retrain-from-scratch re-learned-head AUC for class {fc}: "
          f"{res['auc']:.3f}  (n={res['n']}, n_pos={res.get('n_pos')})")
    print(f"[oracle] -> this is the AUC FLOOR. roar's post-unlearning AUC should be "
          f"compared against THIS, not against 0.5.")
    out = os.path.join(args.output_dir, 'attack_baseline.json')
    with open(out, 'w') as fh:
        json.dump({'forget_class': fc, 'retain_classes': retain,
                   'oracle_relearn_auc': res['auc'], 'n': res['n']}, fh, indent=2)
    print(f"[oracle] wrote {out}")


if __name__ == '__main__':
    main()
