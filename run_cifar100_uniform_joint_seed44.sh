#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
MATRIX_ROOT=/home/chase/Yangxx/VF-CL/results/external_baselines_20260723_214415
FORMAL_OUT=/home/chase/Yangxx/VF-CL/results/formal_cifar100_metrics_20260728
JOB=cifar100:proto_uniform:44
MIN_FREE_MIB=3500
MIN_DISK_KB=$((4 * 1024 * 1024))
LOG="$MATRIX_ROOT/logs/cifar100_proto_uniform_seed44.log"

print_command() {
  "$PY" "$ROOT/external_baseline_matrix.py" command \
    --job "$JOB" --matrix-root "$MATRIX_ROOT" \
    --repo-root "$ROOT" --python "$PY"
}

check_environment() {
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = codex/power-safe-2080
  test -z "$(git -C "$ROOT" status --porcelain)"
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" \
    -ge "$MIN_DISK_KB"
  test -d /home/chase/Yangxx/VF-CL/data/cifar-100-python
  "$PY" -m unittest -q \
    test_formal_cifar100_metrics.py \
    test_external_baseline_matrix.py \
    test_runner_bic.py
}

refresh_formal_table() {
  "$PY" "$ROOT/formal_cifar100_metrics.py" \
    --external-root "$MATRIX_ROOT/runs" \
    --output-dir "$FORMAL_OUT"
}

if test "${1:-}" = --print-command; then
  print_command
  exit 0
fi

if test "${1:-}" = --check; then
  check_environment
  bash -n "$0"
  print_command | grep -q -- "--seed"
  print_command | grep -q -- "44"
  echo CIFAR100_PROTO_UNIFORM_SEED44_CHECK_SUCCESS
  exit 0
fi

check_environment
mkdir -p "$MATRIX_ROOT/logs" "$MATRIX_ROOT/runs"

if RUN_DIR=$("$PY" "$ROOT/external_baseline_matrix.py" find \
    --job "$JOB" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
  RECORDED_COMMIT=$("$PY" "$ROOT/external_baseline_matrix.py" provenance \
    --job "$JOB" --run-dir "$RUN_DIR")
  "$PY" "$ROOT/external_baseline_matrix.py" audit \
    --job "$JOB" --run-dir "$RUN_DIR" \
    --code-commit "$RECORDED_COMMIT" >>"$LOG" 2>&1
  refresh_formal_table >>"$LOG" 2>&1
  touch "$MATRIX_ROOT/CIFAR100_PROTO_UNIFORM_SEED44_SUCCESS"
  echo CIFAR100_PROTO_UNIFORM_SEED44_SUCCESS
  exit 0
fi

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
  awk -F, -v minimum="$MIN_FREE_MIB" \
    '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) >= (minimum + 0)) print $1,$2}' |
  sort -k2,2nr | head -1 | awk '{print $1}')
test -n "$GPU"

mapfile -t CMD < <(print_command)
CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
echo "[$(date --iso-8601=seconds)] start $JOB on physical GPU $GPU" >"$LOG"
if ! CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=44 \
  "${CMD[@]}" >>"$LOG" 2>&1; then
  touch "$MATRIX_ROOT/CIFAR100_PROTO_UNIFORM_SEED44_STOPPED"
  exit 1
fi

RUN_DIR=$("$PY" "$ROOT/external_baseline_matrix.py" find \
  --job "$JOB" --matrix-root "$MATRIX_ROOT")
"$PY" "$ROOT/external_baseline_matrix.py" audit \
  --job "$JOB" --run-dir "$RUN_DIR" \
  --code-commit "$CODE_COMMIT" >>"$LOG" 2>&1
refresh_formal_table >>"$LOG" 2>&1
grep -Fq "Ours (Uniform KD + Joint Cal.),44" \
  "$FORMAL_OUT/FORMAL_CIFAR100_PER_RUN.csv"
touch "$MATRIX_ROOT/CIFAR100_PROTO_UNIFORM_SEED44_SUCCESS"
echo CIFAR100_PROTO_UNIFORM_SEED44_SUCCESS
