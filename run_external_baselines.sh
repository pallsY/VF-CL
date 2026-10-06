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
DATA=$REPO/data
RESULTS=$REPO/results
MIN_DISK_KB=$((4 * 1024 * 1024))
MIN_FREE_MIB=3500

mapfile -t JOBS < <("$PY" "$ROOT/external_baseline_matrix.py" jobs)

if test "${1:-}" = --print-jobs; then
  printf '%s\n' "${JOBS[@]}"
  exit 0
fi

check_environment() {
  test -x "$PY"
  test "$(git -C "$ROOT" branch --show-current)" = codex/power-safe-2080
  test -z "$(git -C "$ROOT" status --porcelain)"
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" -ge "$MIN_DISK_KB"
  test -d "$DATA/cifar-100-python"
  test -f "$DATA/tiny-imagenet-200/DOWNLOAD_SHA256.txt"
  "$PY" "$ROOT/prepare_tinyimagenet.py" --data-path "$DATA" >/dev/null
  test "${#JOBS[@]}" -eq 32
  test "$(printf '%s\n' "${JOBS[@]}" | sort -u | wc -l)" -eq 32
  (cd "$ROOT" && "$PY" -m unittest test_external_baseline_matrix test_runner_bic -q)
}

choose_gpus() {
  while true; do
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits | awk -F, -v minimum="$MIN_FREE_MIB" \
      '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) >= (minimum + 0)) print $1,$2}' |
      sort -k2,2nr | head -2 | awk '{print $1}')
    if test "${#GPUS[@]}" -ge 1; then
      return
    fi
    echo "waiting for a GPU with >=${MIN_FREE_MIB} MiB free"
    sleep 30
  done
}

wait_gpu() {
  local gpu=$1 free
  while true; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d ' ')
    if test "$free" -ge "$MIN_FREE_MIB"; then
      return
    fi
    echo "[$(date --iso-8601=seconds)] waiting for GPU $gpu (${free} MiB free)"
    sleep 30
  done
}

run_worker() {
  local gpu=$1 job run_dir log recorded_commit
  while job=$("$PY" "$ROOT/external_baseline_matrix.py" claim \
    --claims-root "$CLAIMS_ROOT"); do
    if test -f "$MATRIX_ROOT/BASELINE_STOPPED"; then
      return 1
    fi
    if run_dir=$("$PY" "$ROOT/external_baseline_matrix.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT" 2>/dev/null); then
      recorded_commit=$("$PY" "$ROOT/external_baseline_matrix.py" provenance \
        --job "$job" --run-dir "$run_dir")
      "$PY" "$ROOT/external_baseline_matrix.py" audit \
        --job "$job" --run-dir "$run_dir" \
        --code-commit "$recorded_commit" >/dev/null
      echo "[$(date --iso-8601=seconds)] resume-skip $job"
      continue
    fi
    test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" -ge "$MIN_DISK_KB"
    wait_gpu "$gpu"
    mapfile -t CMD < <("$PY" "$ROOT/external_baseline_matrix.py" command \
      --job "$job" --matrix-root "$MATRIX_ROOT" --repo-root "$ROOT" --python "$PY")
    log="$MATRIX_ROOT/logs/${job//:/_}.log"
    echo "[$(date --iso-8601=seconds)] start $job on physical GPU $gpu"
    if ! CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="${job##*:}" \
      "${CMD[@]}" >"$log" 2>&1; then
      printf '%s\n' "$job" >"$MATRIX_ROOT/FAILED_JOB"
      touch "$MATRIX_ROOT/BASELINE_STOPPED"
      return 1
    fi
    run_dir=$("$PY" "$ROOT/external_baseline_matrix.py" find \
      --job "$job" --matrix-root "$MATRIX_ROOT")
    if ! "$PY" "$ROOT/external_baseline_matrix.py" audit \
      --job "$job" --run-dir "$run_dir" --code-commit "$CODE_COMMIT" \
      >>"$log" 2>&1; then
      printf '%s\n' "$job" >"$MATRIX_ROOT/FAILED_JOB"
      touch "$MATRIX_ROOT/BASELINE_STOPPED"
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] complete $job"
  done
}

if test "${1:-}" = --check; then
  check_environment
  bash -n "$ROOT/run_external_baselines.sh"
  echo EXTERNAL_BASELINE_CHECK_SUCCESS
  exit 0
fi

check_environment
choose_gpus
MATRIX_ROOT=${1:-$RESULTS/external_baselines_$(date +%Y%m%d_%H%M%S)}
test ! -e "$MATRIX_ROOT" || test -f "$MATRIX_ROOT/CODE_COMMIT.txt"
mkdir -p "$MATRIX_ROOT/logs" "$MATRIX_ROOT/runs"
CURRENT_CODE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
if test ! -f "$MATRIX_ROOT/CODE_COMMIT.txt"; then
  printf '%s\n' "$CURRENT_CODE_COMMIT" >"$MATRIX_ROOT/CODE_COMMIT.txt"
fi
printf '%s\n' "$(cat "$MATRIX_ROOT/CODE_COMMIT.txt")" \
  >>"$MATRIX_ROOT/CODE_COMMITS.txt"
printf '%s\n' "$CURRENT_CODE_COMMIT" >>"$MATRIX_ROOT/CODE_COMMITS.txt"
sort -u -o "$MATRIX_ROOT/CODE_COMMITS.txt" "$MATRIX_ROOT/CODE_COMMITS.txt"
CODE_COMMIT=$CURRENT_CODE_COMMIT
printf '%s\n' "$MATRIX_ROOT" >"$RESULTS/LATEST_EXTERNAL_BASELINE_ROOT"
printf '%s\n' "${JOBS[@]}" >"$MATRIX_ROOT/JOBS.txt"
CLAIMS_ROOT="$MATRIX_ROOT/claims/$(date +%Y%m%d_%H%M%S)_$$"
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
if test "$status" -ne 0 || test -f "$MATRIX_ROOT/BASELINE_STOPPED"; then
  touch "$MATRIX_ROOT/BASELINE_STOPPED"
  echo BASELINE_STOPPED
  exit 1
fi

"$PY" "$ROOT/external_baseline_matrix.py" summarize --matrix-root "$MATRIX_ROOT"
touch "$MATRIX_ROOT/BASELINE_SUCCESS"
echo BASELINE_SUCCESS
echo "$MATRIX_ROOT"
