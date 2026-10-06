#!/usr/bin/env bash
set -euo pipefail

W=/home/chase/Yangxx/VF-CL/.worktrees/fair-main-table-3datasets
PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
ROOT=${1:-/home/chase/Yangxx/VF-CL/results/fair_main_table_3datasets_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ROOT/logs" "$ROOT/claims"
exec > >(tee -a "$ROOT/launcher.log") 2>&1
echo "[$(date --iso-8601=seconds)] fair main table started root=$ROOT"
git -C "$W" rev-parse HEAD > "$ROOT/CODE_COMMIT.txt"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" "$W/fair_main_table_3datasets.py" check
"$PY" "$W/fair_main_table_3datasets.py" audit-cifar --matrix-root "$ROOT"

if [[ ! -f /home/chase/Yangxx/VF-CL/data/isolet/isolet_vfl.npz ]]; then
  "$PY" "$W/prepare_fair_datasets.py" --dataset isolet | tee -a "$ROOT/logs/preprocess_isolet.log"
fi
if [[ ! -f /home/chase/Yangxx/VF-CL/data/upmc_food101/upmc_food101_vfl.npz ]]; then
  "$PY" "$W/prepare_fair_datasets.py" --dataset upmc --device cuda:0 --batch-size 256 --workers 4 \
    | tee -a "$ROOT/logs/preprocess_upmc.log"
fi

"$PY" "$W/fair_main_table_3datasets.py" run-job isolet:finetune:42 \
  --device cuda:0 --matrix-root "$ROOT" --smoke
"$PY" "$W/fair_main_table_3datasets.py" run-job upmc_food101:finetune:42 \
  --device cuda:0 --matrix-root "$ROOT" --smoke
touch "$ROOT/SMOKE_SUCCESS"

worker() {
  local worker_id=$1
  local device="cuda:$worker_id"
  local spec
  while IFS= read -r spec; do
    [[ -n "$spec" ]] || continue
    echo "[$(date --iso-8601=seconds)] worker=$worker_id starting $spec"
    if ! "$PY" "$W/fair_main_table_3datasets.py" run-job "$spec" \
      --device "$device" --matrix-root "$ROOT"; then
      echo "$spec" > "$ROOT/FAILED_JOB"
      touch "$ROOT/STOPPED"
      return 1
    fi
  done < <("$PY" "$W/fair_main_table_3datasets.py" jobs --worker "$worker_id" --workers 2)
}

worker 0 > "$ROOT/logs/worker_0.log" 2>&1 &
p0=$!
worker 1 > "$ROOT/logs/worker_1.log" 2>&1 &
p1=$!
printf '%s\n' "$p0" > "$ROOT/WORKER_0.pid"
printf '%s\n' "$p1" > "$ROOT/WORKER_1.pid"

status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [[ "$status" -ne 0 || -f "$ROOT/STOPPED" ]]; then
  echo "[$(date --iso-8601=seconds)] queue stopped after a failed job"
  exit 1
fi

"$PY" "$W/fair_main_table_3datasets.py" summarize --matrix-root "$ROOT"
touch "$ROOT/NEW_DATASETS_FAIR_TABLE_SUCCESS"
echo "[$(date --iso-8601=seconds)] fair main table completed"
