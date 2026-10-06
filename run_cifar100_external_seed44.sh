#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
MATRIX_ROOT=/home/chase/Yangxx/VF-CL/results/external_baselines_20260723_214415
FORMAL_OUT=/home/chase/Yangxx/VF-CL/results/formal_cifar100_metrics_20260728
MIN_DISK_KB=$((4 * 1024 * 1024))
MIN_FREE_MIB=3500
STOPPED="$MATRIX_ROOT/CIFAR100_EXTERNAL_SEED44_STOPPED"
SUCCESS="$MATRIX_ROOT/CIFAR100_EXTERNAL_SEED44_SUCCESS"
COMMIT_FILE="$MATRIX_ROOT/CIFAR100_EXTERNAL_SEED44_CODE_COMMIT.txt"

mapfile -t JOBS < <("$PY" "$ROOT/external_baseline_matrix.py" seed44-jobs)

if test "${1:-}" = --print-jobs; then
  printf '%s\n' "${JOBS[@]}"
  exit 0
fi

check_environment() {
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = codex/power-safe-2080
  test -z "$(git -C "$ROOT" status --porcelain)"
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
    -ge "$MIN_DISK_KB"
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  test "${#JOBS[@]}" -eq 7
  test "$(printf '%s\n' "${JOBS[@]}" | sort -u | wc -l)" -eq 7
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
  printf '%s\n' "$1" >"$MATRIX_ROOT/CIFAR100_EXTERNAL_SEED44_FAILED_JOB"
  touch "$STOPPED"
}

run_worker() {
  local gpu=$1 job run_dir log recorded_commit
  local -a command
  while job=$("$PY" "$ROOT/external_baseline_matrix.py" seed44-claim \
    --claims-root "$CLAIMS_ROOT"); do
    test ! -f "$STOPPED"
    if run_dir=$("$PY" "$ROOT/external_baseline_matrix.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
      if test -f "$run_dir/protocol_digest.json"; then
        recorded_commit=$("$PY" "$ROOT/external_baseline_matrix.py" provenance \
          --job "$job" --run-dir "$run_dir")
      else
        recorded_commit=$CODE_COMMIT
      fi
      if ! "$PY" "$ROOT/external_baseline_matrix.py" audit \
          --job "$job" --run-dir "$run_dir" \
          --code-commit "$recorded_commit" >/dev/null; then
        fail_job "$job"
        return 1
      fi
      continue
    fi
    test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
      -ge "$MIN_DISK_KB"
    wait_gpu "$gpu"
    mapfile -t command < <(
      "$PY" "$ROOT/external_baseline_matrix.py" command \
        --job "$job" --matrix-root "$MATRIX_ROOT" \
        --repo-root "$ROOT" --python "$PY"
    )
    if run_dir=$("$PY" "$ROOT/external_baseline_matrix.py" find-incomplete \
      --job "$job" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
      command+=(--resume_run_dir "$run_dir")
    fi
    log="$MATRIX_ROOT/logs/${job//:/_}.log"
    printf '[%s] start/resume %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" >>"$log"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=44 \
      "${command[@]}" >>"$log" 2>&1; then
      fail_job "$job"
      return 1
    fi
    run_dir=$("$PY" "$ROOT/external_baseline_matrix.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT")
    if ! "$PY" "$ROOT/external_baseline_matrix.py" audit \
        --job "$job" --run-dir "$run_dir" \
        --code-commit "$CODE_COMMIT" >>"$log" 2>&1; then
      fail_job "$job"
      return 1
    fi
  done
}

if test "${1:-}" = --check; then
  check_environment
  bash -n "$0"
  "$PY" -m unittest -q \
    test_runner_resume test_resume_method_state test_external_baseline_matrix
  echo CIFAR100_EXTERNAL_SEED44_CHECK_SUCCESS
  exit 0
fi

check_environment
test ! -f "$STOPPED"
mkdir -p "$MATRIX_ROOT/logs" "$MATRIX_ROOT/runs"
CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
if test -f "$COMMIT_FILE"; then
  test "$(cat "$COMMIT_FILE")" = "$CODE_COMMIT"
else
  printf '%s\n' "$CODE_COMMIT" >"$COMMIT_FILE"
fi
choose_gpus
CLAIMS_ROOT="$MATRIX_ROOT/claims/seed44_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$CLAIMS_ROOT"

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" >"$MATRIX_ROOT/seed44_worker_gpu${gpu}.log" 2>&1 &
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

"$PY" "$ROOT/formal_cifar100_metrics.py" \
  --external-root "$MATRIX_ROOT/runs" \
  --output-dir "$FORMAL_OUT" \
  --expected-seeds 42,43,44
touch "$SUCCESS"
echo CIFAR100_EXTERNAL_SEED44_SUCCESS
