#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
DATA=/home/c3080/YangXiaoXiang/VF-CL/data
RESULTS=/home/c3080/YangXiaoXiang/VF-CL/results
P1_ROOT="$RESULTS/bic_p1_formal_20260722_015822"
MIN_DISK_KB=$((4 * 1024 * 1024))
MIN_FREE_MIB=3500

run_dirs() {
  find "$P1_ROOT" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'bic_static_seed42_*' -o -name 'bic_static_seed43_*' \
       -o -name 'bic_uniform_seed42_*' -o -name 'bic_uniform_seed43_*' \) | sort
}

check_environment() {
  test -x "$PY"
  test -d "$DATA/cifar-100-python"
  test "$(git -C "$ROOT" branch --show-current)" = codex/p0-output-bias
  git -C "$ROOT" diff --quiet
  git -C "$ROOT" diff --cached --quiet
  test "$(run_dirs | wc -l)" -eq 4
  while read -r run_dir; do
    test -f "$run_dir/config.json"
    test -f "$run_dir/results.json"
    test -f "$run_dir/final_probs.npz"
    test -f "$run_dir/checkpoints/event_9_CIL.pt"
    test -f "$run_dir/bic/calibration_manifest.json"
    test -f "$run_dir/bic/calibrator.pt"
  done < <(run_dirs)
  test "$(run_dirs | xargs -I{} sha256sum '{}/bic/calibration_manifest.json' | awk '{print $1}' | sort -u | wc -l)" -eq 1
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" -ge "$MIN_DISK_KB"
  test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 1
  (cd "$ROOT" && "$PY" -m unittest test_bic_calibration.py \
    test_offline_calibration_ablation.py test_select_p2_calibration.py \
    test_calibration_split.py -q)
}

choose_gpu() {
  local gpu
  while true; do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F, -v minimum="$MIN_FREE_MIB" \
      '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= minimum) {print $1; exit}}')
    if test -n "$gpu"; then
      echo "$gpu"
      return
    fi
    echo "waiting for any GPU with >=${MIN_FREE_MIB} MiB free" >&2
    sleep 30
  done
}

if test "${1:-}" = --check; then
  check_environment
  echo "P2 offline calibration check passed"
  exit 0
fi

check_environment
P2_ROOT="${1:-$RESULTS/p2_offline_calibration_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$P2_ROOT"
mkdir -p "$P2_ROOT"
git -C "$ROOT" rev-parse HEAD >"$P2_ROOT/CODE_COMMIT.txt"

while read -r run_dir; do
  test "$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')" -ge "$MIN_DISK_KB" || {
    touch "$P2_ROOT/P2_STOPPED"
    echo "stopped: free disk fell below 4 GiB" >&2
    exit 0
  }
  variant=$($PY -c "import json; print(json.load(open('$run_dir/config.json'))['expected_party_kd_variant'])")
  seed=$($PY -c "import json; print(json.load(open('$run_dir/config.json'))['seed'])")
  key="${variant}_seed${seed}"
  output="$P2_ROOT/$key"
  mkdir -p "$output"
  gpu=$(choose_gpu)
  echo "[$(date --iso-8601=seconds)] replaying $key on physical GPU $gpu"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" -u "$ROOT/offline_calibration_ablation.py" \
      --run-dir "$run_dir" --data-path "$DATA" --output-dir "$output" \
      --device cuda:0 >"$output/replay.log" 2>&1
  echo "[$(date --iso-8601=seconds)] evaluating $key on cached logits"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -u "$ROOT/offline_calibration_ablation.py" \
    --run-dir "$run_dir" --cache "$output/logits.npz" \
    --output-json "$output/ablation.json" >"$output/ablation.log" 2>&1
done < <(run_dirs)

"$PY" "$ROOT/select_p2_calibration.py" --root "$P2_ROOT" >"$P2_ROOT/selection.log" 2>&1
test -f "$P2_ROOT/P2_SUCCESS" -o -f "$P2_ROOT/P2_STOPPED"
echo "$P2_ROOT"
