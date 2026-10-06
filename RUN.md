# VF-CUL Patch — Phase 0 + Phase 1 Step 1

Apply this patch on top of your existing `VF-CUL-main` codebase, then follow the steps below.

---

## What changed (summary)

**Phase 0 — Infrastructure**

1. **`cl_methods/er.py`** *(new)* — Experience Replay baseline with reservoir buffer (`er_per_class=20`/class). Serves as a middle-ground reference between FineTune (lower bound) and Oracle (upper bound).

2. **`cl_methods/__init__.py`** — Registered ER under name `'er'`.

3. **`runner.py`** — Oracle now actually trains to convergence:
   - epochs ≥ 200 (was 100)
   - SGD with LR = 0.1 (was using `args.lr` = 1e-3 — severely undertrained)
   - Cosine annealing LR schedule
   - Target: CIFAR-10 ≥ 0.85, CIFAR-100 ≥ 0.60

4. **`main.py`** — Multi-seed support and mean±std aggregation:
   - `--seeds 42,43,44` runs all seeds and aggregates
   - `benchmark_summary.json` reports mean ± std for every cell
   - Adds ER and CL-only baselines (`<cl> x retrain`) to the default `--run_all` matrix

5. **`metrics.py`** — Separated CL metrics:
   - `AA_cil`: mean AA across CIL events only (true CL performance)
   - `AA_ul`: mean AA across UL events only (post-unlearning utility)
   - `AA_final`: AA after the final event
   - BWT now strictly between consecutive CIL events on the intersection of evaluated tasks

6. **`config.py`** — Default `replay_mode` flipped to `'prototype'` (real CL); added `--seeds`, `--oracle_lr`, `--er_per_class`, `--er_batch`. CL method choices now include `'er'`.

7. **`sanity_check.py`** *(new)* — One-shot diagnostic: train each CL method on task 0 only and check accuracy is above a sane threshold. Catches basic implementation bugs.

**Phase 1 Step 1 — proto_evolve FIM fix**

8. **`cl_methods/proto_evolve.py`** — FIM mask now **accumulates across tasks** (OR-merge) instead of being reset every task. Previously, the mask was wiped at every task, so old-task parameter protection was lost the moment a new task started — likely the root cause of V-LETO's collapse at task 3+.

---

## Setup

```bash
# Replace VF-CUL-main with the patched version, then:
cd VF-CUL-main
pip install -r requirements.txt
```

Verify environment:
```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Execution plan (4090, CIFAR-10)

We do this in **three steps**, each with a clear success criterion. **Stop and ping me with logs at each ✋ STOP.**

### Step A: Sanity check (~3-5 min)

Verify every CL method can learn task 0. If any method fails this, there's a bug to fix before going further.

```bash
python sanity_check.py \
  --data cifar10 --num_classes 10 \
  --num_tasks 5 --classes_per_task 2 \
  --num_parties 2 --aggregation sum --model_type resnet18 \
  --epochs_per_task 20 \
  --batch_size 128 --lr 0.01 \
  --device cuda:0 --seed 42 \
  --results_dir ./results
```

**Expected**: All 7 methods (finetune, er, proto_aug, proto_evolve, proto_fedspace, der_pp, er_ace) reach task_0 acc ≥ 0.85.

✋ **STOP A**: paste the final summary table and `./results/sanity_cifar10.json` back to me.
- If all pass → continue
- If anything fails → we debug before continuing

---

### Step B: Tune Oracle (~30-60 min)

Confirm the upper bound is high enough to be a meaningful target. Single seed, single combo, just Oracle.

```bash
python main.py \
  --cl_method finetune --ul_method retrain \
  --data cifar10 --num_classes 10 \
  --num_tasks 5 --classes_per_task 2 \
  --unlearn_after_tasks 99,99 --unlearn_classes "0;0" \
  --num_parties 2 --aggregation sum --model_type resnet18 \
  --epochs_per_task 30 --batch_size 128 --lr 0.01 \
  --replay_mode prototype \
  --device cuda:0 --seed 42 \
  --results_dir ./results
```

Wait — actually for Step B I want to run **only Oracle**. The above runs FineTune. To run just Oracle, easiest way is `--run_all` then take Oracle from the summary. Or use this minimal Oracle-only invocation: open a Python REPL and call `run_oracle(args)` directly. For simplicity, just use `--run_all` (longer but gives full table):

Skip Step B as standalone — it's covered by Step C.

---

### Step C: Phase 1 first dry run (~1.5-2 h on 4090)

Run the full benchmark, 3 seeds, on CIFAR-10. Includes:
- Oracle (upper bound)
- FineTune (lower bound)
- ER (middle reference)
- All CL × UL combos including the patched proto_evolve

```bash
python main.py --run_all \
  --data cifar10 --num_classes 10 \
  --num_tasks 5 --classes_per_task 2 \
  --unlearn_after_tasks 2,3 --unlearn_classes "0;4" \
  --num_parties 2 --aggregation sum --model_type resnet18 \
  --epochs_per_task 30 --ul_epochs 5 \
  --batch_size 128 --lr 0.01 --ul_lr 1e-4 \
  --replay_mode prototype \
  --seeds 42,43,44 \
  --device cuda:0 \
  --results_dir ./results
```

**Expected outcome** (these are predictions — please report what you actually get):

| Method | AA_cil | AA_final | Notes |
|---|---|---|---|
| FineTune (CL only) | ≈ 0.30 | ≈ 0.20 | Lower bound, complete forgetting |
| ER (CL only) | ≈ 0.55-0.65 | ≈ 0.50 | Should beat FineTune by 25+ pts |
| V-LETO (proto_evolve, CL only) — **fixed** | ≈ 0.55-0.70 | ≈ 0.50-0.60 | Should NOT collapse at task 3+ anymore |
| Oracle | ≈ 0.80-0.85 | ≈ 0.80 | Real upper bound now |

The critical things I want to see:

1. **Oracle AA_final ≥ 0.75** — confirms my Oracle fix actually worked
2. **ER significantly above FineTune** — confirms ER baseline is sane
3. **V-LETO no longer collapses** at later tasks — confirms FIM accumulation fix worked
4. **PASS+VFL and FedSpace+VFL still likely broken** — we expect this, will fix in Phase 1 Step 2

✋ **STOP C**: paste back to me:
1. The console summary table (last 20 lines of stdout)
2. `./results/full_benchmark_<timestamp>/benchmark_summary.json`
3. Any error tracebacks

---

## Troubleshooting

**`CUDA out of memory`**: lower `--batch_size` to 64 or 32.

**`download failed` for CIFAR**: the dataset will auto-download to `./data` first run; needs internet. If your 4090 box is offline, copy `cifar-10-batches-py/` from another machine.

**Training looks weirdly slow**: ResNet18 on 4090 with batch 128 should do ≈ 5-10 sec/epoch on CIFAR-10. If it's 30+ sec/epoch, check `num_workers` and that the data isn't being loaded from a slow disk.

**Sanity check fails for some method**: paste the full error or the per-method accuracy. The most likely culprits I expect:
- `er_ace`: in `prototype` mode there are no old-class samples in a batch, so the ACE branch never triggers — task 0 accuracy should still be fine (it falls through to standard CE) but later tasks won't have CL protection
- `der_pp`: should work for task 0 (no old model yet) but later tasks need the missing buffer

---

## Files to send back at each STOP

At STOP A: `./results/sanity_cifar10.json` + the printed summary table.

At STOP C: `./results/full_benchmark_<timestamp>/benchmark_summary.json` + the printed summary table at end of stdout.

For both: also any traceback if something crashed.
