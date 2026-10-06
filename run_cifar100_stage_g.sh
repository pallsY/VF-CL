#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536
root=$(cd "$(dirname "$0")" && pwd)
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python
base=/home/chase/Yangxx/VF-CL/results/cifar100_aa_final_improvement_20260805_085859
study="$base/stage_g_supcon"
stage_a="$study/stage_a"
stage_b="$study/stage_b"
min_disk_kb=$((8 * 1024 * 1024))
min_free_mib=3500
max_gpu_util=70
stopped="$study/STOPPED"

if test "${1:-}" = --check; then
  bash -n "$0"
  "$python" -m unittest -q \
    test_cifar100_stage_g test_cifar100_stage_f \
    test_proto_evolve_balanced_replay test_resume_method_state test_config_resume \
    test_cifar100_stage_e_training test_cifar100_stage_e_b \
    test_cifar100_stage_c_head test_offline_balanced_head \
    test_feature_retention_validation test_runner_resume
  echo CIFAR100_STAGE_G_CHECK_SUCCESS
  exit 0
fi

check_environment() {
  test -x "$python"
  test "$(git -C "$root" branch --show-current)" = codex/cifar100-aa-final-improvement
  test -z "$(git -C "$root" status --porcelain)"
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  test "$(df --output=avail -k "$root" | tail -1 | tr -d ' ')" -ge "$min_disk_kb"
  mapfile -t jobs < <("$python" "$root/cifar100_stage_g.py" jobs-a)
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

claim_job() {
  local phase=$1 claims=$2
  if test "$phase" = a; then
    "$python" "$root/cifar100_stage_g.py" claim-a --claims-root "$claims"
  else
    "$python" "$root/cifar100_stage_g.py" claim-b \
      --stage-a-summary "$stage_a/STAGE_G_A_SUMMARY.json" --claims-root "$claims"
  fi
}

run_worker() {
  local gpu=$1 phase=$2 phase_root=$3 claims=$4
  local job run_dir log seed head_dir head_json
  local -a command
  while true; do
    wait_gpu "$gpu"
    job=$(claim_job "$phase" "$claims")
    test -n "$job" || return 0
    seed=${job##*:}
    log="$phase_root/logs/${job//:/_}.log"
    head_dir="$phase_root/heads/${job//:/_}"
    head_json="$head_dir/validation.json"
    if ! run_dir=$("$python" "$root/cifar100_stage_g.py" find \
        --job "$job" --study-root "$phase_root" 2>/dev/null); then
      mapfile -t command < <(
        "$python" "$root/cifar100_stage_g.py" command \
          --job "$job" --study-root "$phase_root" --repo-root "$root" --python "$python"
      )
      if run_dir=$("$python" "$root/cifar100_stage_g.py" find-incomplete \
          --job "$job" --study-root "$phase_root" 2>/dev/null); then
        command+=(--resume_run_dir "$run_dir")
      fi
      printf '[%s] start/resume %s on physical GPU %s\n' \
        "$(date --iso-8601=seconds)" "$job" "$gpu" >> "$log"
      if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
          OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="$seed" \
          "${command[@]}" >> "$log" 2>&1; then
        fail_job "$phase:$job:train"
        return 1
      fi
      run_dir=$("$python" "$root/cifar100_stage_g.py" find \
        --job "$job" --study-root "$phase_root")
    fi
    if test ! -s "$head_json"; then
      mkdir -p "$head_dir/artifacts"
      if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
          OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="$seed" \
          "$python" "$root/cifar100_stage_c_head.py" validation \
          --run-dir "$run_dir" --output-dir "$head_dir/artifacts" \
          --output-json "$head_json" --device cuda:0 >> "$log" 2>&1; then
        fail_job "$phase:$job:head"
        return 1
      fi
    fi
    if ! "$python" "$root/cifar100_stage_g.py" audit \
        --job "$job" --run-dir "$run_dir" --head-json "$head_json" \
        --code-commit "$code_commit" >> "$log" 2>&1; then
      fail_job "$phase:$job:audit"
      return 1
    fi
  done
}

run_phase() {
  local phase=$1 phase_root=$2 claims status=0
  mkdir -p "$phase_root/logs" "$phase_root/runs" "$phase_root/heads"
  claims="$phase_root/claims/$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$claims"
  pids=()
  for gpu in "${gpus[@]}"; do
    run_worker "$gpu" "$phase" "$phase_root" "$claims" \
      > "$phase_root/worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if test "$status" -ne 0 || test -f "$stopped"; then
    fail_job "stage-g-$phase"
    return 1
  fi
}

check_environment
mkdir -p "$study"
test ! -f "$stopped"
test ! -f "$study/STAGE_G_SUCCESS"
code_commit=$(git -C "$root" rev-parse HEAD)
printf '%s\n' "$code_commit" > "$study/CODE_COMMIT.txt"
mapfile -t gpus < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    awk '{gsub(/ /, ""); print}' | head -2
)
test "${#gpus[@]}" -eq 2

run_phase a "$stage_a"
"$python" "$root/cifar100_stage_g.py" summarize-a \
  --study-root "$stage_a" --code-commit "$code_commit" \
  > "$stage_a/summary.log" 2>&1
touch "$stage_a/STAGE_G_A_SUCCESS"

mapfile -t promoted_jobs < <(
  "$python" "$root/cifar100_stage_g.py" jobs-b \
    --stage-a-summary "$stage_a/STAGE_G_A_SUMMARY.json"
)
if test "${#promoted_jobs[@]}" -eq 0 || test -z "${promoted_jobs[0]:-}"; then
  touch "$study/STAGE_G_NO_CANDIDATE"
  echo CIFAR100_STAGE_G_NO_CANDIDATE
  exit 0
fi

run_phase b "$stage_b"
"$python" "$root/cifar100_stage_g.py" summarize-b \
  --stage-a-summary "$stage_a/STAGE_G_A_SUMMARY.json" \
  --study-root "$stage_b" --code-commit "$code_commit" \
  > "$stage_b/summary.log" 2>&1
touch "$stage_b/STAGE_G_B_SUCCESS"
if "$python" - "$stage_b/STAGE_G_SELECTION.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)
PY
then
  touch "$study/STAGE_G_SUCCESS"
  echo CIFAR100_STAGE_G_SUCCESS
else
  touch "$study/STAGE_G_NO_CANDIDATE"
  echo CIFAR100_STAGE_G_NO_CANDIDATE
fi
