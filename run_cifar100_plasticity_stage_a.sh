#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

root=$(cd "$(dirname "$0")" && pwd)
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python
study=/home/chase/Yangxx/VF-CL/results/cifar100_aa_final_improvement_20260805_085859/plasticity_stage_a
min_disk_kb=$((8 * 1024 * 1024))
min_free_mib=3500
max_gpu_util=70
stopped="$study/STOPPED"
success="$study/STAGE_A_SUCCESS"

check_environment() {
  test -x "$python"
  test "$(git -C "$root" branch --show-current)" = codex/cifar100-aa-final-improvement
  test -z "$(git -C "$root" status --porcelain)"
  test ! -e /proc/mounts || ! grep -qE '[[:space:]]/brf[[:space:]]' /proc/mounts
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  test "$(df --output=avail -k "$root" | tail -1 | tr -d ' ')" -ge "$min_disk_kb"
  mapfile -t jobs < <("$python" "$root/cifar100_plasticity_validation.py" jobs)
  test "${#jobs[@]}" -eq 4
  test "$(printf '%s\n' "${jobs[@]}" | sort -u | wc -l)" -eq 4
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
  local gpu=$1 claims=$2 job run_dir log
  local -a command
  while true; do
    wait_gpu "$gpu"
    job=$("$python" "$root/cifar100_plasticity_validation.py" claim --claims-root "$claims")
    test -n "$job" || return 0
    if run_dir=$("$python" "$root/cifar100_plasticity_validation.py" find \
        --job "$job" --study-root "$study" 2>/dev/null); then
      "$python" "$root/cifar100_plasticity_validation.py" audit \
        --job "$job" --run-dir "$run_dir" --code-commit "$code_commit" >/dev/null
      continue
    fi
    mapfile -t command < <(
      "$python" "$root/cifar100_plasticity_validation.py" command \
        --job "$job" --study-root "$study" --repo-root "$root" --python "$python"
    )
    if run_dir=$("$python" "$root/cifar100_plasticity_validation.py" find-incomplete \
        --job "$job" --study-root "$study" 2>/dev/null); then
      command+=(--resume_run_dir "$run_dir")
    fi
    log="$study/logs/${job//:/_}.log"
    printf '[%s] start/resume %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" >> "$log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
        "${command[@]}" >> "$log" 2>&1; then
      fail_job "$job:train"
      return 1
    fi
    run_dir=$("$python" "$root/cifar100_plasticity_validation.py" find \
      --job "$job" --study-root "$study")
    if ! "$python" "$root/cifar100_plasticity_validation.py" audit \
        --job "$job" --run-dir "$run_dir" --code-commit "$code_commit" \
        >> "$log" 2>&1; then
      fail_job "$job:audit"
      return 1
    fi
  done
}

if test "${1:-}" = --check; then
  bash -n "$0"
  "$python" -m unittest -q \
    test_cifar100_plasticity_validation test_offline_balanced_head \
    test_feature_retention_validation test_runner_resume
  echo CIFAR100_PLASTICITY_STAGE_A_CHECK_SUCCESS
  exit 0
fi

check_environment
mkdir -p "$study/logs" "$study/runs"
test ! -f "$stopped"
test ! -f "$success"
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
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if test "$status" -ne 0 || test -f "$stopped"; then
  fail_job stage-a
  exit 1
fi

"$python" "$root/cifar100_plasticity_validation.py" summarize \
  --study-root "$study" --code-commit "$code_commit" --bwt-floor -0.20 \
  > "$study/summary.log" 2>&1
touch "$success"
echo CIFAR100_PLASTICITY_STAGE_A_SUCCESS
