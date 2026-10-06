#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
STUDY_ROOT=${VFCL_FEATURE_ROOT:-/home/chase/Yangxx/VF-CL/results/cifar100_feature_retention_$(date +%Y%m%d_%H%M%S)}
REFERENCE_ROOT=${VFCL_REFERENCE_ROOT:-/home/chase/Yangxx/VF-CL/results/cifar100_party_kd_lambda_validation_20260729_114147}
EXTERNAL_ROOT=${VFCL_EXTERNAL_ROOT:-/home/chase/Yangxx/VF-CL/results/external_baselines_20260723_214415}
SELECTION_OUT="$STUDY_ROOT/selection"
MIN_DISK_KB=$((8 * 1024 * 1024))
MIN_FREE_MIB=3500
MAX_GPU_UTIL=70
STOPPED="$STUDY_ROOT/STOPPED"
NO_CANDIDATE="$STUDY_ROOT/FEATURE_RETENTION_NO_CANDIDATE"
SUCCESS="$STUDY_ROOT/FEATURE_RETENTION_FORMAL_SUCCESS"
COMMIT_FILE="$STUDY_ROOT/CODE_COMMIT.txt"

mapfile -t STAGE_A_JOBS < <(
  "$PY" "$ROOT/feature_retention_validation.py" jobs \
    --stage A --selection-dir "$SELECTION_OUT"
)

if test "${1:-}" = --print-jobs; then
  printf '%s\n' "${STAGE_A_JOBS[@]}"
  exit 0
fi

check_disk() {
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
    -ge "$MIN_DISK_KB"
}

check_environment() {
  local frozen
  frozen=(
    cl_methods/proto_evolve.py config.py runner.py data_utils.py
    bic_calibration.py calibration_split.py formal_cifar100_metrics.py
  )
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = \
    codex/cifar100-feature-retention
  test -z "$(git -C "$ROOT" status --porcelain)"
  check_disk
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  test -f "$REFERENCE_ROOT/VALIDATION_SUCCESS"
  test -d "$EXTERNAL_ROOT"
  git -C "$ROOT" diff --quiet 8d63af5 HEAD -- "${frozen[@]}"
  test "${#STAGE_A_JOBS[@]}" -eq 3
  test "$(printf '%s\n' "${STAGE_A_JOBS[@]}" | sort -u | wc -l)" -eq 3
}

choose_gpus() {
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits |
      awk '{gsub(/ /, ""); print}' | head -2
  )
  test "${#GPUS[@]}" -ge 1
}

wait_gpu() {
  local gpu=$1 free util
  while true; do
    test ! -f "$STOPPED"
    read -r free util < <(
      nvidia-smi --id="$gpu" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1,$2}'
    )
    if test "$free" -ge "$MIN_FREE_MIB" && \
        test "$util" -le "$MAX_GPU_UTIL"; then
      return 0
    fi
    sleep 30
  done
}

fail_job() {
  mkdir -p "$STUDY_ROOT"
  printf '%s\n' "$1" >"$STUDY_ROOT/FAILED_JOB"
  touch "$STOPPED"
}

run_worker() {
  local stage=$1 mode=$2 claims_root=$3 gpu=$4 job run_dir log
  local -a command
  while true; do
    wait_gpu "$gpu"
    if ! job=$("$PY" "$ROOT/feature_retention_validation.py" claim \
        --stage "$stage" --claims-root "$claims_root" \
        --selection-dir "$SELECTION_OUT"); then
      return 0
    fi
    test ! -f "$STOPPED"
    if run_dir=$("$PY" "$ROOT/feature_retention_validation.py" find \
        --job "$job" --mode "$mode" --study-root "$STUDY_ROOT" 2>/dev/null); then
      test -s "$run_dir/checkpoints/event_9_CIL.pt"
      if ! "$PY" "$ROOT/feature_retention_validation.py" audit \
          --job "$job" --mode "$mode" --run-dir "$run_dir" \
          --code-commit "$CODE_COMMIT" >/dev/null; then
        fail_job "$stage:$job:audit-existing"
        return 1
      fi
      continue
    fi
    if ! check_disk; then
      fail_job "$stage:$job:disk"
      return 1
    fi
    mapfile -t command < <(
      "$PY" "$ROOT/feature_retention_validation.py" command \
        --job "$job" --mode "$mode" --study-root "$STUDY_ROOT" \
        --repo-root "$ROOT" --python "$PY"
    )
    if run_dir=$("$PY" "$ROOT/feature_retention_validation.py" find-incomplete \
        --job "$job" --mode "$mode" --study-root "$STUDY_ROOT" 2>/dev/null); then
      command+=(--resume_run_dir "$run_dir")
    fi
    log="$STUDY_ROOT/logs/${stage}_${job//:/_}.log"
    printf '[%s] start/resume %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" >>"$log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="${job##*:}" \
        "${command[@]}" >>"$log" 2>&1; then
      fail_job "$stage:$job:train"
      return 1
    fi
    run_dir=$("$PY" "$ROOT/feature_retention_validation.py" find \
      --job "$job" --mode "$mode" --study-root "$STUDY_ROOT")
    if ! "$PY" "$ROOT/feature_retention_validation.py" audit \
        --job "$job" --mode "$mode" --run-dir "$run_dir" \
        --code-commit "$CODE_COMMIT" >>"$log" 2>&1; then
      fail_job "$stage:$job:audit-new"
      return 1
    fi
  done
}

run_stage() {
  local stage=$1 mode=$2 claims_root status pid gpu
  local -a pids
  claims_root="$STUDY_ROOT/claims/${stage}_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$claims_root"
  pids=()
  for gpu in "${GPUS[@]}"; do
    run_worker "$stage" "$mode" "$claims_root" "$gpu" \
      >"$STUDY_ROOT/worker_${stage}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if test "$status" -ne 0 || test -f "$STOPPED"; then
    touch "$STOPPED"
    return 1
  fi
}

if test "${1:-}" = --check; then
  check_environment
  bash -n "$0"
  "$PY" -m unittest -q \
    test_feature_retention_validation test_feature_retention_pipeline \
    test_party_kd_lambda_validation test_formal_cifar100_metrics \
    test_runner_resume
  echo CIFAR100_FEATURE_RETENTION_CHECK_SUCCESS
  exit 0
fi

check_environment
mkdir -p "$STUDY_ROOT/logs" "$STUDY_ROOT/validation/runs" \
  "$STUDY_ROOT/formal/runs" "$SELECTION_OUT"
test ! -f "$STOPPED"
test ! -f "$NO_CANDIDATE"
test ! -f "$SUCCESS"
CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
if test -f "$COMMIT_FILE"; then
  test "$(cat "$COMMIT_FILE")" = "$CODE_COMMIT"
else
  printf '%s\n' "$CODE_COMMIT" >"$COMMIT_FILE"
fi
choose_gpus

if ! run_stage A validation; then
  fail_job stage-a
  exit 1
fi
if ! "$PY" "$ROOT/feature_retention_validation.py" select-stage-a \
    --study-root "$STUDY_ROOT" --reference-root "$REFERENCE_ROOT" \
    --code-commit "$CODE_COMMIT" >"$STUDY_ROOT/stage_a_selection.log" 2>&1; then
  fail_job stage-a-selection
  exit 1
fi
if test -f "$NO_CANDIDATE"; then
  echo FEATURE_RETENTION_NO_CANDIDATE
  exit 0
fi
test -s "$SELECTION_OUT/PROMOTED_WEIGHTS"

if ! run_stage B validation; then
  fail_job stage-b
  exit 1
fi
if ! "$PY" "$ROOT/feature_retention_validation.py" select-stage-b \
    --study-root "$STUDY_ROOT" --reference-root "$REFERENCE_ROOT" \
    --code-commit "$CODE_COMMIT" >"$STUDY_ROOT/stage_b_selection.log" 2>&1; then
  fail_job stage-b-selection
  exit 1
fi
if test -f "$NO_CANDIDATE"; then
  echo FEATURE_RETENTION_NO_CANDIDATE
  exit 0
fi
test -s "$SELECTION_OUT/SELECTED_FEAT_DISTILL_WEIGHT"

if ! run_stage formal formal; then
  fail_job formal
  exit 1
fi
if ! "$PY" "$ROOT/feature_retention_validation.py" report-formal \
    --study-root "$STUDY_ROOT" --external-root "$EXTERNAL_ROOT" \
    --code-commit "$CODE_COMMIT" >"$STUDY_ROOT/formal_report.log" 2>&1; then
  fail_job formal-report
  exit 1
fi
touch "$SUCCESS"
echo FEATURE_RETENTION_FORMAL_SUCCESS
