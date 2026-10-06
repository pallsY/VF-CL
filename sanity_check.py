"""Sanity check: each CL method on task 0 only.

A correctly implemented CL method (even FineTune) should reach reasonable
accuracy on the FIRST task — because there is no forgetting to deal with
yet. This script catches the most basic implementation bugs (broken forward,
wrong output dim, training loop errors).

Pass criteria:
  CIFAR-10  (2 classes per task):  task_0 acc >= 0.85
  CIFAR-100 (20 classes per task): task_0 acc >= 0.50

If a method fails this, do NOT proceed to full benchmark — there's a bug.

Usage:
    python sanity_check.py --data cifar10 --epochs_per_task 20 --device cuda:0
"""
import os, sys, json, time, copy
import torch
import numpy as np
from datetime import datetime
from config import get_config
from data_utils import TaskManager, VFLDataset
from models import build_models
from vfl_trainer import VFLTrainer
from cl_methods import get_cl_method
from utils_logging import tee_to_file

METHODS_TO_TEST = ['finetune', 'er', 'proto_aug', 'proto_evolve',
                   'proto_fedspace', 'der_pp', 'er_ace']


def run_sanity(args, cl_name):
    """Train cl_name on task 0 ONLY and evaluate."""
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    task_mgr = TaskManager(args)
    timeline = task_mgr.get_timeline()

    # Find the very first CIL event (task 0)
    first_cil = next((e for e in timeline if e['type'] == 'CIL'), None)
    assert first_cil is not None, "No CIL event in timeline?"

    tid = first_cil['task_id']
    new_cls = first_cil['new_classes']
    task_mgr.advance_task(tid)

    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    cl_method = get_cl_method(cl_name, trainer, args)

    print(f"\n  [{cl_name}] task {tid}, classes {new_cls}, epochs={args.epochs_per_task}")
    cl_method.before_task(tid, new_cls, task_mgr.get_effective_classes())

    if hasattr(cl_method, 'before_train_compute_pre_protos'):
        task_loader, _ = dataset.get_task_loaders(new_cls)
        cl_method.before_train_compute_pre_protos(task_loader)

    train_loader, _ = dataset.get_task_loaders(new_cls)
    t0 = time.time()
    history, elapsed = cl_method.train_task(train_loader, tid)
    cl_method.after_task(train_loader, tid)

    _, test_loader = dataset.get_task_loaders(new_cls, shuffle_train=False)
    acc, _, _ = trainer.evaluate(test_loader)
    print(f"  [{cl_name}] task_0 test acc = {acc:.4f}   time = {elapsed:.1f}s")
    return acc, elapsed


def main():
    args = get_config()
    # Disable UL events for sanity check
    args.unlearn_after_tasks = [99, 99]
    args.unlearn_classes = [[0], [0]]

    threshold = 0.85 if args.data == 'cifar10' else 0.50

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.results_dir, f'sanity_{args.data}_{ts}.log')

    with tee_to_file(log_path):
        print(f"\n{'='*70}")
        print(f"  SANITY CHECK: task 0 accuracy threshold = {threshold:.2f}")
        print(f"  data={args.data}  num_parties={args.num_parties}  agg={args.aggregation}  model={args.model_type}")
        print(f"  epochs_per_task={args.epochs_per_task}  lr={args.lr}  batch={args.batch_size}")
        print(f"{'='*70}")

        results = {}
        for m in METHODS_TO_TEST:
            try:
                m_args = copy.deepcopy(args)
                acc, elapsed = run_sanity(m_args, m)
                results[m] = {'acc': round(float(acc), 4), 'time': round(float(elapsed), 1),
                              'pass': bool(acc >= threshold)}
            except Exception as e:
                import traceback; traceback.print_exc()
                results[m] = {'error': str(e), 'pass': False}

        print(f"\n{'='*70}")
        print(f"  SANITY CHECK SUMMARY ({args.data}, threshold = {threshold:.2f})")
        print(f"{'='*70}")
        print(f"  {'Method':<20} {'Acc':>8} {'Time(s)':>10} {'Status':>10}")
        print('-' * 56)
        n_pass = 0
        for m in METHODS_TO_TEST:
            r = results.get(m, {})
            if 'error' in r:
                print(f"  {m:<20} {'ERR':>8} {'-':>10} {'FAIL':>10}")
                continue
            status = 'PASS' if r['pass'] else 'FAIL'
            n_pass += 1 if r['pass'] else 0
            print(f"  {m:<20} {r['acc']:>8.4f} {r['time']:>10.1f} {status:>10}")
        print('-' * 56)
        print(f"  {n_pass}/{len(METHODS_TO_TEST)} methods passed.\n")

        out = os.path.join(args.results_dir, f'sanity_{args.data}.json')
        with open(out, 'w') as f:
            json.dump({'threshold': threshold, 'results': results, 'config': vars(args),
                       'timestamp': ts, 'log_path': log_path},
                      f, indent=2, default=str)
        print(f"  Results JSON: {out}")
        print(f"  Run log:      {log_path}")

    # Exit code reflects pass/fail for shell integration
    sys.exit(0 if n_pass == len(METHODS_TO_TEST) else 1)


if __name__ == '__main__':
    main()
