"""Direct test of 'centralized + bad => code bug'.

Holds EVERYTHING constant vs the LwF run (1 party = centralized, cosine head,
same trainer/head/eval, same 30-epoch budget) and changes ONE thing:
class-incremental sequential  ->  JOINT training on all 10 classes at once.

If the code were broken, joint training would also collapse. If joint reaches
~0.9 on every 2-class group, the code is sound and the earlier class-IL=0 is
purely the incremental-no-replay regime, not a bug.
"""
import torch, numpy as np, time
from config import get_config
from data_utils import VFLDataset, split_features
from models import build_models
from vfl_trainer import VFLTrainer


@torch.no_grad()
def eval_classes(trainer, loader, args):
    for b in trainer.bottoms:
        b.eval()
    trainer.top_model.eval()
    c = t = 0
    for x, y in loader:
        x = x.to(args.device); y = y.to(args.device)
        parts = split_features(x, args)
        embs = [trainer.bottoms[i](parts[i]) for i in range(len(trainer.bottoms))]
        out = trainer.top_model(trainer._aggregate(embs))
        c += (out.argmax(1) == y).sum().item(); t += y.size(0)
    return c / max(t, 1)


def main():
    args = get_config()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)

    allc = list(range(10))
    train_l, _ = dataset.get_task_loaders(allc)
    t0 = time.time()
    trainer.train_task(train_l, args.epochs_per_task)
    print(f"JOINT trained 10 classes in {time.time()-t0:.0f}s "
          f"(num_parties={args.num_parties}, cosine={getattr(args,'cosine_head',False)})", flush=True)

    print("\n=== JOINT (1-party centralized) per 2-class group, full-head argmax ===")
    pairs = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}
    for t, cs in pairs.items():
        _, test_l = dataset.get_task_loaders(cs)
        print(f"  classes {cs}: class-IL acc = {eval_classes(trainer, test_l, args):.3f}")
    _, all_test = dataset.get_task_loaders(allc)
    print(f"  ALL 10 classes overall = {eval_classes(trainer, all_test, args):.3f}")


if __name__ == '__main__':
    main()
