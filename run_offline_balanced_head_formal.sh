#!/usr/bin/env bash
set -euo pipefail

repo=/home/chase/Yangxx/VF-CL
worktree="$repo/.worktrees/cifar100-aa-final-improvement"
study="$repo/results/cifar100_feature_retention_validation_20260801_011547"
root="$repo/results/cifar100_aa_final_improvement_20260805_085859"
validation="$root/offline_head_cv"
output="$root/formal_head"
python=/home/chase/anaconda3/envs/mlz_3.9/bin/python

run42="$study/formal/runs/cifar100_feature_retention_formal_feat_0p05_seed42_20260802_183130"
run43="$study/formal/runs/cifar100_feature_retention_formal_feat_0p05_seed43_20260802_183130"
run44="$study/formal/runs/cifar100_feature_retention_formal_feat_0p05_seed44_20260802_195523"
selection="$validation/SELECTION.json"

mkdir -p "$output/seed42" "$output/seed43" "$output/seed44"
cd "$worktree"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py formal \
  --run-dir "$run42" --output-dir "$output/seed42/artifacts" \
  --selection "$selection" --output-json "$output/seed42/formal.json" \
  --device cuda:0 > "$output/seed42/run.log" 2>&1 &
pid42=$!

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=43 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py formal \
  --run-dir "$run43" --output-dir "$output/seed43/artifacts" \
  --selection "$selection" --output-json "$output/seed43/formal.json" \
  --device cuda:0 > "$output/seed43/run.log" 2>&1 &
pid43=$!

wait "$pid42"
wait "$pid43"

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=44 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  offline_balanced_head.py formal \
  --run-dir "$run44" --output-dir "$output/seed44/artifacts" \
  --selection "$selection" --output-json "$output/seed44/formal.json" \
  --device cuda:0 > "$output/seed44/run.log" 2>&1

"$python" offline_balanced_head.py summarize \
  --records "$output/seed42/formal.json" "$output/seed43/formal.json" "$output/seed44/formal.json" \
  --selection "$selection" --output-json "$output/FORMAL_SUMMARY.json" \
  > "$output/summary.log" 2>&1

touch "$output/FORMAL_SUCCESS"
