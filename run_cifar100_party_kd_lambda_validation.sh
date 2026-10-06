#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

die() {
  printf 'FATAL: %s\n' "$*" >&2
  return 1
}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
: "${VFCL_PYTHON:?VFCL_PYTHON must name the reviewed Python interpreter}"
PY=$(realpath -e -- "$VFCL_PYTHON") || \
  { die "VFCL_PYTHON does not exist"; exit 1; }
[[ -x "$PY" ]] || { die "VFCL_PYTHON is not executable"; exit 1; }
COMMON=$(git -C "$ROOT" rev-parse --git-common-dir) || \
  { die "cannot derive git common directory"; exit 1; }
[[ "$COMMON" = /* ]] || COMMON="$ROOT/$COMMON"
COMMON=$(realpath -e -- "$COMMON") || \
  { die "git common directory does not exist"; exit 1; }
[[ "$(basename -- "$COMMON")" == .git ]] || \
  { die "invalid git common directory"; exit 1; }
REPO=$(dirname -- "$COMMON")
"$PY" -c 'import pathlib,sys; raise SystemExit(pathlib.Path(sys.executable).resolve()!=pathlib.Path(sys.argv[1]).resolve())' "$PY" || \
  { die "VFCL_PYTHON does not identify its own interpreter"; exit 1; }
MATRIX_ROOT=${VFCL_LAMBDA_MATRIX_ROOT:-$REPO/results/cifar100_party_kd_lambda_validation_$(date +%Y%m%d_%H%M%S)}
MIN_DISK_KB=$((8 * 1024 * 1024))
MIN_FREE_MIB=3500
STOPPED="$MATRIX_ROOT/STOPPED"
SUCCESS="$MATRIX_ROOT/VALIDATION_SUCCESS"
COMMIT_FILE="$MATRIX_ROOT/CODE_COMMIT.txt"
SELECTION_OUT="$MATRIX_ROOT/selection"

mapfile -t JOBS < <("$PY" "$ROOT/party_kd_lambda_validation.py" jobs)

if test "${1:-}" = --print-jobs; then
  printf '%s\n' "${JOBS[@]}"
  exit 0
fi

check_environment() {
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = \
    codex/cifar100-lambda-validation
  test -z "$(git -C "$ROOT" status --porcelain)"
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
    -ge "$MIN_DISK_KB"
  test -d "$REPO/data/cifar-100-python"
  test "${#JOBS[@]}" -eq 15
  test "$(printf '%s\n' "${JOBS[@]}" | sort -u | wc -l)" -eq 15
  mapfile -t probe < <(
    "$PY" "$ROOT/party_kd_lambda_validation.py" command \
      --job lambda_0.25:42 --matrix-root "$MATRIX_ROOT" \
      --repo-root "$ROOT" --python "$PY"
  )
  printf '%s ' "${probe[@]}" | grep -q -- '--save_task_checkpoints 3'
  printf '%s ' "${probe[@]}" | grep -q -- '--lambda_validation_enabled 1'
}

choose_gpus() {
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F, -v minimum="$MIN_FREE_MIB" \
        '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) >= (minimum + 0)) print $1,$2}' |
      sort -k2,2nr | head -2 | awk '{print $1}'
  )
  test "${#GPUS[@]}" -ge 1
}

wait_gpu() {
  local gpu=$1 free
  while true; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d ' ')
    if test "$free" -ge "$MIN_FREE_MIB"; then
      return
    fi
    sleep 30
  done
}

fail_job() {
  printf '%s\n' "$1" >"$MATRIX_ROOT/FAILED_JOB"
  touch "$STOPPED"
}

run_worker() {
  local gpu=$1 job run_dir log
  local -a command
  while job=$("$PY" "$ROOT/party_kd_lambda_validation.py" claim \
    --claims-root "$CLAIMS_ROOT"); do
    test ! -f "$STOPPED"
    if run_dir=$("$PY" "$ROOT/party_kd_lambda_validation.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
      if ! "$PY" "$ROOT/party_kd_lambda_validation.py" audit \
          --job "$job" --run-dir "$run_dir" \
          --code-commit "$CODE_COMMIT" >/dev/null; then
        fail_job "$job"
        return 1
      fi
      test -f "$run_dir/protocol_digest.json"
      continue
    fi
    test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
      -ge "$MIN_DISK_KB"
    wait_gpu "$gpu"
    mapfile -t command < <(
      "$PY" "$ROOT/party_kd_lambda_validation.py" command \
        --job "$job" --matrix-root "$MATRIX_ROOT" \
        --repo-root "$ROOT" --python "$PY"
    )
    if run_dir=$("$PY" "$ROOT/party_kd_lambda_validation.py" find-incomplete \
      --job "$job" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
      command+=(--resume_run_dir "$run_dir")
    fi
    log="$MATRIX_ROOT/logs/${job//:/_}.log"
    printf '[%s] start/resume %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" >>"$log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="${job##*:}" \
      "${command[@]}" >>"$log" 2>&1; then
      fail_job "$job"
      return 1
    fi
    run_dir=$("$PY" "$ROOT/party_kd_lambda_validation.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT")
    if ! "$PY" "$ROOT/party_kd_lambda_validation.py" audit \
        --job "$job" --run-dir "$run_dir" \
        --code-commit "$CODE_COMMIT" >>"$log" 2>&1; then
      fail_job "$job"
      return 1
    fi
    test -f "$run_dir/protocol_digest.json"
  done
}

if test "${1:-}" = --check; then
  check_environment
  bash -n "$0"
  "$PY" -m unittest -q \
    test_calibration_split test_party_kd_lambda_validation \
    test_party_kd_lambda_pipeline test_runner_resume
  echo CIFAR100_PARTY_KD_LAMBDA_CHECK_SUCCESS
  exit 0
fi

check_environment
mkdir -p "$MATRIX_ROOT/logs" "$MATRIX_ROOT/runs"
test ! -f "$STOPPED"
test ! -f "$SUCCESS"
CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
if test -f "$COMMIT_FILE"; then
  test "$(cat "$COMMIT_FILE")" = "$CODE_COMMIT"
else
  printf '%s\n' "$CODE_COMMIT" >"$COMMIT_FILE"
fi
choose_gpus
CLAIMS_ROOT="$MATRIX_ROOT/claims/validation_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$CLAIMS_ROOT"

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" >"$MATRIX_ROOT/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if test "$status" -ne 0 || test -f "$STOPPED"; then
  touch "$STOPPED"
  exit 1
fi

if ! "$PY" "$ROOT/party_kd_lambda_validation.py" select \
    --matrix-root "$MATRIX_ROOT" --output-dir "$SELECTION_OUT" \
    --code-commit "$CODE_COMMIT" >"$MATRIX_ROOT/selection.log" 2>&1; then
  fail_job selection
  exit 1
fi
touch "$SUCCESS"
echo CIFAR100_PARTY_KD_LAMBDA_VALIDATION_SUCCESS
