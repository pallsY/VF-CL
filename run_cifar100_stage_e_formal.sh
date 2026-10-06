#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536
root=$(cd "$(dirname "$0")" && pwd)
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python
base=/home/chase/Yangxx/VF-CL/results/cifar100_aa_final_improvement_20260805_085859
training_selection="$base/stage_e_training_stage_b/STAGE_E_B_SELECTION.json"
head_selection="$base/stage_e_head_validation/SELECTION.json"
study="$base/stage_e_formal"
min_disk_kb=$((8 * 1024 * 1024))
min_free_mib=3500
max_gpu_util=70
stopped="$study/STOPPED"

if test "${1:-}" = --check; then
  bash -n "$0"
  "$python" -m unittest -q \
    test_cifar100_stage_e_formal test_cifar100_stage_e_b \
    test_cifar100_stage_e_training test_proto_evolve_balanced_replay \
    test_resume_method_state test_config_resume test_cifar100_stage_c_head \
    test_offline_balanced_head test_cifar100_plasticity_formal \
    test_cifar100_plasticity_stage_b test_cifar100_plasticity_validation \
    test_feature_retention_validation test_runner_resume
  echo CIFAR100_STAGE_E_FORMAL_CHECK_SUCCESS
  exit 0
fi

check_environment() {
  test -x "$python"
  test -s "$training_selection"
  test -s "$head_selection"
  test -f "$base/stage_e_training_stage_b/STAGE_E_B_SUCCESS"
  test -f "$base/stage_e_head_validation/VALIDATION_SUCCESS"
  test "$(git -C "$root" branch --show-current)" = \
    codex/cifar100-aa-final-improvement
  test -z "$(git -C "$root" status --porcelain)"
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  test "$(df --output=avail -k "$root" | tail -1 | tr -d ' ')" \
    -ge "$min_disk_kb"
  mapfile -t jobs < <(
    "$python" "$root/cifar100_stage_e_formal.py" jobs \
      --selection "$training_selection"
  )
  test "${#jobs[@]}" -eq 3
}

wait_gpu() {
  local gpu=$1 free util
  while true; do
    test ! -f "$stopped"
    read -r free util < <(
      nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | \
        awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1,$2}'
    )
    if test "$free" -ge "$min_free_mib" && test "$util" -le "$max_gpu_util"; then
      return 0
    fi
    sleep 30
  done
}

fail_job() {
  mkdir -p "$study"
  printf '%s\n' "$1" > "$study/FAILED_JOB"
  touch "$stopped"
}

run_worker() {
  local gpu=$1 claims=$2 job run_dir log seed
  local -a command
  while true; do
    wait_gpu "$gpu"
    job=$("$python" "$root/cifar100_stage_e_formal.py" claim \
      --selection "$training_selection" --claims-root "$claims")
    test -n "$job" || return 0
    seed=${job##*:}
    if run_dir=$("$python" "$root/cifar100_stage_e_formal.py" find \
        --job "$job" --selection "$training_selection" \
        --study-root "$study" 2>/dev/null); then
      "$python" "$root/cifar100_stage_e_formal.py" audit \
        --job "$job" --selection "$training_selection" --run-dir "$run_dir" \
        --code-commit "$code_commit" >/dev/null
      continue
    fi
    mapfile -t command < <(
      "$python" "$root/cifar100_stage_e_formal.py" command \
        --job "$job" --selection "$training_selection" \
        --study-root "$study" --repo-root "$root" --python "$python"
    )
    if run_dir=$("$python" "$root/cifar100_stage_e_formal.py" find-incomplete \
        --job "$job" --selection "$training_selection" \
        --study-root "$study" 2>/dev/null); then
      command+=(--resume_run_dir "$run_dir")
    fi
    log="$study/logs/${job//:/_}.log"
    printf '[%s] start/resume %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" >> "$log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="$seed" \
        "${command[@]}" >> "$log" 2>&1; then
      fail_job "$job:train"
      return 1
    fi
    run_dir=$("$python" "$root/cifar100_stage_e_formal.py" find \
      --job "$job" --selection "$training_selection" --study-root "$study")
    if ! "$python" "$root/cifar100_stage_e_formal.py" audit \
        --job "$job" --selection "$training_selection" --run-dir "$run_dir" \
        --code-commit "$code_commit" >> "$log" 2>&1; then
      fail_job "$job:audit"
      return 1
    fi
  done
}

check_environment
mkdir -p "$study/logs" "$study/runs" "$study/balanced_head"
test ! -f "$stopped"
test ! -f "$study/FORMAL_SUCCESS"
code_commit=$(git -C "$root" rev-parse HEAD)
printf '%s\n' "$code_commit" > "$study/CODE_COMMIT.txt"
claims="$study/claims/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$claims"
mapfile -t gpus < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    awk '{gsub(/ /, ""); print}' | head -2
)
pids=()
for gpu in "${gpus[@]}"; do
  run_worker "$gpu" "$claims" > "$study/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if test "$status" -ne 0 || test -f "$stopped"; then
  fail_job stage-e-formal-training
  exit 1
fi

mapfile -t jobs < <(
  "$python" "$root/cifar100_stage_e_formal.py" jobs \
    --selection "$training_selection"
)
formal_runs=()
for job in "${jobs[@]}"; do
  formal_runs+=("$("$python" "$root/cifar100_stage_e_formal.py" find \
    --job "$job" --selection "$training_selection" --study-root "$study")")
done

pids=()
for index in 0 1; do
  seed=$((42 + index))
  CUDA_VISIBLE_DEVICES="$index" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED="$seed" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$python" "$root/cifar100_stage_c_head.py" formal \
    --run-dir "${formal_runs[$index]}" \
    --output-dir "$study/balanced_head/seed${seed}/artifacts" \
    --selection "$head_selection" \
    --output-json "$study/balanced_head/seed${seed}/formal.json" \
    --device cuda:0 > "$study/balanced_head/seed${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=44 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  "$root/cifar100_stage_c_head.py" formal \
  --run-dir "${formal_runs[2]}" \
  --output-dir "$study/balanced_head/seed44/artifacts" \
  --selection "$head_selection" \
  --output-json "$study/balanced_head/seed44/formal.json" \
  --device cuda:0 > "$study/balanced_head/seed44.log" 2>&1

"$python" "$root/cifar100_stage_c_head.py" summarize \
  --records "$study/balanced_head/seed42/formal.json" \
  "$study/balanced_head/seed43/formal.json" \
  "$study/balanced_head/seed44/formal.json" \
  --selection "$head_selection" --output-json "$study/FORMAL_SUMMARY.json" \
  > "$study/formal_summary.log" 2>&1
touch "$study/FORMAL_SUCCESS"
echo CIFAR100_STAGE_E_FORMAL_SUCCESS
