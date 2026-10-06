#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
DATA=/home/c3080/YangXiaoXiang/VF-CL/data
RESULTS=/home/c3080/YangXiaoXiang/VF-CL/results
MIN_DISK_KB=$((4 * 1024 * 1024))
MIN_FREE_MIB=3500

check_environment() {
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = codex/p0-output-bias
  git -C "$ROOT" diff --quiet
  git -C "$ROOT" diff --cached --quiet
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" -ge "$MIN_DISK_KB"
  test -f "$DATA/tiny-imagenet-200/DOWNLOAD_SHA256.txt"
  "$PY" "$ROOT/prepare_tinyimagenet.py" --data-path "$DATA" >/dev/null
  test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 1
  (cd "$ROOT" && "$PY" -m unittest test_bic_calibration.py \
    test_calibration_split.py test_runner_bic.py test_prepare_tinyimagenet.py \
    test_summarize_p3.py -q)
}

choose_gpu() {
  while true; do
    local gpu
    gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | awk -F, -v minimum="$MIN_FREE_MIB" \
      '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= minimum) print $1,$3,$2}' |
      sort -k2,2n -k3,3nr | awk 'NR==1 {print $1}')
    if test -n "$gpu"; then
      echo "$gpu"
      return
    fi
    echo "waiting for any GPU with >=${MIN_FREE_MIB} MiB free" >&2
    sleep 30
  done
}

run_seed() {
  local seed=$1 p3_root=$2 gpu name log run_dir
  gpu=$(choose_gpu)
  name="p3_tiny_uniform_seed${seed}"
  log="$p3_root/${name}.log"
  echo "[$(date --iso-8601=seconds)] starting $name on physical GPU $gpu" >&2
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u "$ROOT/main.py" \
      --cl_method proto_evolve --ul_method retrain --replay_mode prototype \
      --model_type resnet18 --num_parties 4 --aggregation sum \
      --data tinyimagenet --data_path "$DATA" --num_classes 200 \
      --num_tasks 10 --classes_per_task 20 --epochs_per_task 50 \
      --batch_size 64 --unlearn_after_tasks 99 --unlearn_classes 0 \
      --results_dir "$p3_root" --exp_name "$name" \
      --dep_tracking_enabled 1 --party_kd_enabled 1 \
      --party_kd_mode uniform --expected_party_kd_variant uniform \
      --party_kd_lambda 1.0 --save_task_checkpoints 2 \
      --bic_enabled 1 --bic_fit_mode joint_final --bic_per_class 25 \
      --bic_split_seed 20260722 --bic_lr 0.05 --bic_steps 1000 \
      --seed "$seed" --device cuda:0 --deterministic 1 --num_workers 2 \
      >"$log" 2>&1
  run_dir=$(find "$p3_root" -mindepth 1 -maxdepth 1 -type d \
    -name "${name}_*" | sort | tail -1)
  test -f "$run_dir/results.json"
  test -f "$run_dir/checkpoints/event_9_CIL.pt"
  echo "$run_dir"
}

if test "${1:-}" = --check; then
  check_environment
  echo "P3 TinyImageNet check passed"
  exit 0
fi

check_environment
P3_ROOT="${1:-$RESULTS/p3_tinyimagenet_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$P3_ROOT"
mkdir -p "$P3_ROOT"
git -C "$ROOT" rev-parse HEAD >"$P3_ROOT/CODE_COMMIT.txt"
cp "$DATA/tiny-imagenet-200/DOWNLOAD_SHA256.txt" "$P3_ROOT/"

seed42=$(run_seed 42 "$P3_ROOT")
"$PY" "$ROOT/summarize_p3.py" --output-dir "$P3_ROOT" \
  --pilot "$seed42/results.json" >"$P3_ROOT/seed42_gate.log"
passed=$($PY -c "import json; print(int(json.load(open('$P3_ROOT/P3_SEED42_GATE.json'))['passed']))")
if test "$passed" != 1; then
  "$PY" "$ROOT/summarize_p3.py" --output-dir "$P3_ROOT" \
    --formal "seed42=$seed42/results.json" >"$P3_ROOT/final_summary.log"
  echo "$P3_ROOT"
  exit 0
fi

seed43=$(run_seed 43 "$P3_ROOT")
"$PY" "$ROOT/summarize_p3.py" --output-dir "$P3_ROOT" --formal \
  "seed42=$seed42/results.json" "seed43=$seed43/results.json" \
  >"$P3_ROOT/final_summary.log"
test -f "$P3_ROOT/P3_SUCCESS" -o -f "$P3_ROOT/P3_STOPPED"
echo "$P3_ROOT"
