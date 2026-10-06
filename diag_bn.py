"""Decisive test: is LwF's class-IL=0 caused by BatchNorm running-stat domination
by the last task, or by genuine representation/weight collapse?

Re-runs the LwF CIL loop (no UL), then evaluates every task's test set TWICE at
the final state:
  (a) BN eval-mode  -> running stats (== the standard results.json numbers)
  (b) BN train-mode -> per-batch stats (BN sees the task-t distribution)
If (b) >> (a) on old tasks, BN running stats are the culprit (implementation-
level amplifier). If (b) ~ (a) ~ 0, the collapse is genuine feature/weight drift.
"""
import torch, numpy as np, time
from config import get_config
from data_utils import VFLDataset, TaskManager, split_features
from models import build_models
from vfl_trainer import VFLTrainer
from cl_methods import get_cl_method


def _set_bn_train(module):
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()


@torch.no_grad()
def eval_task(trainer, loader, args, bn_train=False):
    for b in trainer.bottoms:
        b.eval()
    trainer.top_model.eval()
    if bn_train:
        for b in trainer.bottoms:
            _set_bn_train(b)
    correct = total = 0
    for x, y in loader:
        x = x.to(args.device); y = y.to(args.device)
        parts = split_features(x, args)
        embs = [trainer.bottoms[i](parts[i]) for i in range(len(trainer.bottoms))]
        agg = trainer._aggregate(embs)
        out = trainer.top_model(agg)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main():
    args = get_config()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    task_mgr = TaskManager(args)
    timeline = task_mgr.get_timeline()
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.dataset_ref = dataset
    cl_method = get_cl_method(args.cl_method, trainer, args)

    task_classes = {}
    for event in timeline:
        if event['type'] != 'CIL':
            continue
        tid = event['task_id']; new_cls = event['new_classes']
        task_mgr.advance_task(tid)
        task_classes[tid] = new_cls
        eff = task_mgr.get_effective_classes()
        task_loader, _ = dataset.get_task_loaders(new_cls)
        cl_method.before_task(tid, new_cls, eff)
        if args.replay_mode == 'full':
            train_loader, _ = dataset.get_task_loaders(eff)
        else:
            train_loader = task_loader
        t0 = time.time()
        cl_method.train_task(train_loader, tid)
        cl_method.after_task(task_loader, tid)
        print(f"[task {tid}] trained classes {new_cls} in {time.time()-t0:.0f}s", flush=True)

    print("\n=== FINAL-STATE per-task class-IL (full-head argmax) ===")
    print(" task | classes | BN-eval(running) | BN-train(batch stats)")
    # do ALL bn-eval first (bn-train mode mutates running stats as a side effect)
    bn_eval = {}
    for t, cs in task_classes.items():
        _, test_l = dataset.get_task_loaders(cs)
        bn_eval[t] = eval_task(trainer, test_l, args, bn_train=False)
    for t, cs in task_classes.items():
        _, test_l = dataset.get_task_loaders(cs)
        a = bn_eval[t]
        b = eval_task(trainer, test_l, args, bn_train=True)
        print(f"  t{t}  | {cs} |      {a:.3f}       |        {b:.3f}")


if __name__ == '__main__':
    main()
