#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
DATA=/home/c3080/YangXiaoXiang/VF-CL/data
RESULTS_BASE=/home/c3080/YangXiaoXiang/VF-CL/results
MIN_DISK_KB=$((6 * 1024 * 1024))
MIN_FREE_MIB=3000
MAX_GPU_UTIL=80

check_environment() {
  test -x "$PY"
  test -d "$DATA"
  test "$(git -C "$ROOT" branch --show-current)" = codex/p0-output-bias
  git -C "$ROOT" diff --quiet
  git -C "$ROOT" diff --cached --quiet
  local available
  available=$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')
  test "$available" -ge "$MIN_DISK_KB"
  test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 1
  "$PY" -m unittest test_calibration_split.py test_bic_calibration.py \
    test_runner_bic.py test_summarize_bic_p1.py -q
}

wait_for_gpu() {
  while true; do
    local gpu
    gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | awk -F, \
      -v min_free="$MIN_FREE_MIB" -v max_util="$MAX_GPU_UTIL" \
      '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= min_free && $3 <= max_util) {print $1; exit}}')
    if test -n "$gpu"; then
      echo "$gpu"
      return
    fi
    echo "waiting for a GPU with >=${MIN_FREE_MIB} MiB free and <=${MAX_GPU_UTIL}% utilization" >&2
    sleep 30
  done
}

run_one() {
  local variant=$1 seed=$2 tasks=$3 epochs=$4 model=$5 result_root=$6
  local gpu name log
  gpu=$(wait_for_gpu)
  name="bic_${variant}_seed${seed}"
  log="$result_root/${name}.log"
  echo "starting $name on physical GPU $gpu" >&2
  (
    cd "$ROOT"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u main.py \
      --cl_method proto_evolve --ul_method retrain --replay_mode prototype \
      --model_type "$model" --num_parties 4 --aggregation sum \
      --data cifar100 --data_path "$DATA" --num_classes 100 \
      --num_tasks "$tasks" --classes_per_task 10 --epochs_per_task "$epochs" \
      --batch_size 64 --unlearn_after_tasks 99 --unlearn_classes 0 \
      --results_dir "$result_root" --exp_name "$name" \
      --dep_tracking_enabled 1 --party_kd_enabled 1 \
      --party_kd_mode "$variant" --expected_party_kd_variant "$variant" \
      --party_kd_lambda 1.0 --save_task_checkpoints 2 \
      --bic_enabled 1 --bic_per_class 25 --bic_split_seed 20260722 \
      --bic_lr 0.05 --bic_steps 200 \
      --seed "$seed" --device cuda:0 --deterministic 1 --num_workers 2
  ) >"$log" 2>&1
  local run_dir
  run_dir=$(find "$result_root" -maxdepth 1 -type d -name "${name}_*" | sort | tail -1)
  test -f "$run_dir/results.json"
  test -f "$run_dir/checkpoints/event_$((tasks - 1))_CIL.pt"
  echo "$run_dir"
}

if test "${1:-}" = --check; then
  check_environment
  echo "P1 pipeline check passed"
  exit 0
fi

if test "${1:-}" = --smoke-then-full; then
  "$0" --smoke
  exec "$0"
fi

check_environment
P1_ROOT="$RESULTS_BASE/bic_p1_formal_$(date +%Y%m%d_%H%M%S)"
if test "${1:-}" = --smoke; then
  P1_ROOT="$RESULTS_BASE/bic_p1_smoke_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$P1_ROOT"
  run_one static 42 2 1 small_cnn "$P1_ROOT" >/dev/null
  touch "$P1_ROOT/SMOKE_SUCCESS"
  echo "$P1_ROOT"
  exit 0
fi

mkdir -p "$P1_ROOT"
git -C "$ROOT" rev-parse HEAD >"$P1_ROOT/CODE_COMMIT.txt"
static42=$(run_one static 42 10 50 resnet18 "$P1_ROOT")
"$PY" "$ROOT/summarize_bic_p1.py" \
  --pilot-results "$static42/results.json" --output-dir "$P1_ROOT"
if test -f "$P1_ROOT/P1_STOPPED"; then
  echo "P1 pilot failed; remaining runs were not launched" >&2
  exit 0
fi

static43=$(run_one static 43 10 50 resnet18 "$P1_ROOT")
uniform42=$(run_one uniform 42 10 50 resnet18 "$P1_ROOT")
uniform43=$(run_one uniform 43 10 50 resnet18 "$P1_ROOT")
"$PY" "$ROOT/summarize_bic_p1.py" --output-dir "$P1_ROOT" --formal-results \
  "static42=$static42/results.json" "static43=$static43/results.json" \
  "uniform42=$uniform42/results.json" "uniform43=$uniform43/results.json"
echo "$P1_ROOT"
