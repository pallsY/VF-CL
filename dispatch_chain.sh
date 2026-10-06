#!/usr/bin/env bash
# Chain the benchmark grids so the 8 GPUs are never double-booked:
#   covtype grid (already running) -> nuswide grid (if npz present) -> cifar100 O1.
# Launch ONCE with nohup; safe to re-run (each dispatcher skips finished combos).
cd /public/home/dongshou/cl_fix2/run

echo "[chain] waiting for covtype grid (ALL_GRIDTAB_DONE.flag) ..."
while [ ! -f ALL_GRIDTAB_DONE.flag ]; do sleep 120; done
echo "[chain] covtype grid done $(date)"

if [ -f data/nuswide_vfl/nuswide_vfl.npz ]; then
  echo "[chain] launching nuswide grid $(date)"
  rm -f ALL_GRIDTAB_DONE.flag
  DATASETS=nuswide ORDERS="1 2" bash dispatch_grid_tab.sh > log_gridtab_nuswide.txt 2>&1
  echo "[chain] nuswide grid done $(date)"
else
  echo "[chain] WARN: nuswide npz missing, skipping (dispatch manually later)"
fi

echo "[chain] launching cifar100 grid O1 $(date)"
ORDERS=1 bash dispatch_grid_cifar.sh > log_gridcifar_o1.txt 2>&1
echo "[chain] ALL DONE $(date)"
