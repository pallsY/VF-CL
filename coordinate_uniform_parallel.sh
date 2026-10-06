#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/c3080/YangXiaoXiang/VF-CL/.worktrees/p0-output-bias
PY=/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
DATA=/home/c3080/YangXiaoXiang/VF-CL/data
FORMAL=/home/c3080/YangXiaoXiang/VF-CL/results/bic_p1_formal_20260722_015822
MASTER_PID=${1:?master pid required}
UNIFORM42_PID=${2:?uniform42 pid required}
UNIFORM43_PID=

verify() {
  ps -p "$MASTER_PID" -o cmd= | grep -q 'run_bic_p1_pipeline.sh'
  ps -p "$UNIFORM42_PID" -o cmd= | grep -q 'bic_uniform_seed42'
  test -z "$(find "$FORMAL" -maxdepth 1 -type d -name 'bic_uniform_seed43_*' -print -quit)"
  test "$(df --output=avail -k "$FORMAL" | tail -1 | tr -d ' ')" -ge $((4 * 1024 * 1024))
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
  test "$free" -ge 3000
}

if test "${3:-}" = --check; then
  verify
  echo 'parallel coordinator check passed'
  exit 0
fi

recover() {
  kill -CONT "$MASTER_PID" 2>/dev/null || true
  if test -n "$UNIFORM43_PID"; then
    kill -TERM "$UNIFORM43_PID" 2>/dev/null || true
  fi
}
trap recover ERR INT TERM

verify
kill -STOP "$MASTER_PID"
echo "paused serial master $MASTER_PID"
(
  cd "$ROOT"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=43 CUDA_VISIBLE_DEVICES=0 \
  "$PY" -u main.py \
    --cl_method proto_evolve --ul_method retrain --replay_mode prototype \
    --model_type resnet18 --num_parties 4 --aggregation sum \
    --data cifar100 --data_path "$DATA" --num_classes 100 \
    --num_tasks 10 --classes_per_task 10 --epochs_per_task 50 \
    --batch_size 64 --unlearn_after_tasks 99 --unlearn_classes 0 \
    --results_dir "$FORMAL" --exp_name bic_uniform_seed43 \
    --dep_tracking_enabled 1 --party_kd_enabled 1 \
    --party_kd_mode uniform --expected_party_kd_variant uniform \
    --party_kd_lambda 1.0 --save_task_checkpoints 2 \
    --bic_enabled 1 --bic_per_class 25 --bic_split_seed 20260722 \
    --bic_lr 0.05 --bic_steps 200 \
    --seed 43 --device cuda:0 --deterministic 1 --num_workers 2
) >"$FORMAL/bic_uniform_seed43.log" 2>&1 &
UNIFORM43_PID=$!
echo "started Uniform-43 on physical GPU 0 as pid $UNIFORM43_PID"

while true; do
  uniform42_results=$(find "$FORMAL" -maxdepth 2 -path '*/bic_uniform_seed42_*/results.json' -print -quit)
  uniform43_results=$(find "$FORMAL" -maxdepth 2 -path '*/bic_uniform_seed43_*/results.json' -print -quit)
  if test -n "$uniform42_results" && test -n "$uniform43_results"; then
    break
  fi
  if test -z "$uniform42_results" && ! kill -0 "$UNIFORM42_PID" 2>/dev/null; then
    echo 'Uniform-42 exited without results.json' >&2
    false
  fi
  if test -z "$uniform43_results" && ! kill -0 "$UNIFORM43_PID" 2>/dev/null; then
    echo 'Uniform-43 exited without results.json' >&2
    false
  fi
  sleep 30
done

kill -TERM -- "-$MASTER_PID" 2>/dev/null || true
trap - ERR INT TERM
static42=$(find "$FORMAL" -maxdepth 2 -path '*/bic_static_seed42_*/results.json' -print -quit)
static43=$(find "$FORMAL" -maxdepth 2 -path '*/bic_static_seed43_*/results.json' -print -quit)
"$PY" "$ROOT/summarize_bic_p1.py" --output-dir "$FORMAL" --formal-results \
  "static42=$static42" "static43=$static43" \
  "uniform42=$uniform42_results" "uniform43=$uniform43_results"
echo "parallel P1 pipeline complete: $FORMAL/P1_FINAL_REPORT.md"
