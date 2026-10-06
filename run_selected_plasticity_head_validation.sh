#!/usr/bin/env bash
set -euo pipefail

repo=/home/chase/Yangxx/VF-CL
worktree="$repo/.worktrees/cifar100-aa-final-improvement"
base="$repo/results/cifar100_aa_final_improvement_20260805_085859"
stage_b="$base/plasticity_stage_b/STAGE_B_SELECTION.json"
output="$base/plasticity_head_validation"
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python

test -s "$stage_b"
mkdir -p "$output/seed42" "$output/seed43" "$output/seed44"
cd "$worktree"

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
candidate=${selected[0]}
run42=${selected[1]}
run43=${selected[2]}
run44=${selected[3]}
printf '%s\n' "$candidate" > "$output/TRAINING_CANDIDATE.txt"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py validation \
  --run-dir "$run42" --output-dir "$output/seed42/artifacts" \
  --output-json "$output/seed42/validation.json" --device cuda:0 \
  > "$output/seed42/run.log" 2>&1 &
pid42=$!

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=43 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py validation \
  --run-dir "$run43" --output-dir "$output/seed43/artifacts" \
  --output-json "$output/seed43/validation.json" --device cuda:0 \
  > "$output/seed43/run.log" 2>&1 &
pid43=$!

wait "$pid42"
wait "$pid43"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=44 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py validation \
  --run-dir "$run44" --output-dir "$output/seed44/artifacts" \
  --output-json "$output/seed44/validation.json" --device cuda:0 \
  > "$output/seed44/run.log" 2>&1

"$python" offline_balanced_head.py select \
  --records "$output/seed42/validation.json" "$output/seed43/validation.json" "$output/seed44/validation.json" \
  --bwt-floor -0.20 --output-json "$output/SELECTION.json" \
  > "$output/selection.log" 2>&1
touch "$output/VALIDATION_SUCCESS"
