#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

repo=/home/chase/Yangxx/VF-CL
root="$repo/.worktrees/cifar100-aa-final-improvement"
base="$repo/results/cifar100_aa_final_improvement_20260805_085859"
stage_b="$base/plasticity_stage_b/STAGE_B_SELECTION.json"
output="$base/stage_c_head_validation"
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python
stopped="$output/STOPPED"

if test "${1:-}" = --check; then
  bash -n "$0"
  "$python" -m unittest -q \
    test_cifar100_stage_c_head test_offline_balanced_head \
    test_cifar100_plasticity_formal test_cifar100_plasticity_stage_b \
    test_cifar100_plasticity_validation test_feature_retention_validation \
    test_runner_resume
  echo CIFAR100_STAGE_C_HEAD_CHECK_SUCCESS
  exit 0
fi

test -x "$python"
test -s "$stage_b"
test -f "$base/plasticity_stage_b/STAGE_B_SUCCESS"
test "$(git -C "$root" branch --show-current)" = \
  codex/cifar100-aa-final-improvement
test -z "$(git -C "$root" status --porcelain)"
test ! -f "$stopped"
test ! -f "$output/VALIDATION_SUCCESS"
mkdir -p "$output/seed42" "$output/seed43" "$output/seed44"
cd "$root"

mapfile -t selected < <(
  "$python" - "$stage_b" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
candidate=d['selected_candidate']
rows=sorted(
    (r for r in d['records'] if r['candidate'] == candidate),
    key=lambda r: int(r['seed']),
)
if [int(r['seed']) for r in rows] != [42, 43, 44]:
    raise SystemExit('selected candidate does not have seeds 42,43,44')
print(candidate)
for row in rows:
    print(row['run_dir'])
PY
)
test "${#selected[@]}" -eq 4
printf '%s\n' "${selected[0]}" > "$output/TRAINING_CANDIDATE.txt"

run_seed() {
  local gpu=$1 seed=$2 run_dir=$3
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED="$seed" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$python" cifar100_stage_c_head.py validation \
    --run-dir "$run_dir" --output-dir "$output/seed${seed}/artifacts" \
    --output-json "$output/seed${seed}/validation.json" --device cuda:0 \
    > "$output/seed${seed}/run.log" 2>&1
}

run_seed 0 42 "${selected[1]}" &
pid42=$!
run_seed 1 43 "${selected[2]}" &
pid43=$!
status=0
wait "$pid42" || status=1
wait "$pid43" || status=1
if test "$status" -ne 0; then
  touch "$stopped"
  exit 1
fi

run_seed 0 44 "${selected[3]}"

"$python" cifar100_stage_c_head.py select \
  --records "$output/seed42/validation.json" \
  "$output/seed43/validation.json" "$output/seed44/validation.json" \
  --bwt-floor -0.15 --output-json "$output/SELECTION.json" \
  > "$output/selection.log" 2>&1
touch "$output/VALIDATION_SUCCESS"
echo CIFAR100_STAGE_C_HEAD_VALIDATION_SUCCESS
