#!/usr/bin/env bash
set -euo pipefail

ulimit -Sn 65536

ROOT=/home/chase/Yangxx/VF-CL/.worktrees/cifar100-feature-retention
PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
: "${SMOKE_ROOT:?SMOKE_ROOT is required}"
: "${STUDY_ROOT:?STUDY_ROOT is required}"
: "${REFERENCE_ROOT:?REFERENCE_ROOT is required}"
: "${EXTERNAL_ROOT:?EXTERNAL_ROOT is required}"
MIN_FREE_MIB=3500
MAX_GPU_UTIL=70

mkdir -p "$SMOKE_ROOT/uninterrupted/runs" "$SMOKE_ROOT/resume/runs"
trap 'touch "$SMOKE_ROOT/SMOKE_FAILED"' ERR

wait_for_gpu() {
  local gpu free util
  while true; do
    while read -r gpu; do
      read -r free util < <(
        nvidia-smi --id="$gpu" \
          --query-gpu=memory.free,utilization.gpu \
          --format=csv,noheader,nounits |
          awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1,$2}'
      )
      if test "$free" -ge "$MIN_FREE_MIB" && \
          test "$util" -le "$MAX_GPU_UTIL"; then
        printf '%s\n' "$gpu"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    sleep 30
  done
}

COMMON=(
  "$ROOT/main.py" --data cifar100 --data_path /home/chase/Yangxx/VF-CL/data
  --num_classes 20 --num_tasks 2 --classes_per_task 10
  --unlearn_after_tasks 99 --unlearn_classes 0
  --num_parties 4 --model_type resnet18 --aggregation sum
  --epochs_per_task 1 --batch_size 64 --num_workers 2
  --lr 0.001 --momentum 0.9 --weight_decay 0.0005
  --cl_method proto_evolve --ul_method retrain --replay_mode prototype
  --deterministic 1 --data_flow_audit 1 --seed 42
  --dep_tracking_enabled 1 --party_kd_enabled 1 --party_kd_mode uniform
  --expected_party_kd_variant uniform --party_kd_lambda 1.0
  --distill_weight 0.5 --proto_lambda_a 0.1 --fim_freeze_frac 0.25
  --proto_sdc true --feat_distill_weight 0.05
  --bic_enabled 1 --bic_per_class 1 --bic_split_seed 20260722
  --bic_lr 0.05 --bic_steps 10 --bic_fit_mode joint_each_stage
  --lambda_validation_enabled 1 --lambda_validation_per_class 1
  --lambda_validation_split_seed 20260729 --save_task_checkpoints 3
)

GPU=$(wait_for_gpu)
printf '%s\n' "$GPU" >"$SMOKE_ROOT/SMOKE_GPU"
printf '[%s] uninterrupted smoke on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  "$PY" "${COMMON[@]}" \
  --results_dir "$SMOKE_ROOT/uninterrupted/runs" \
  --exp_name feature_retention_smoke_uninterrupted \
  >"$SMOKE_ROOT/uninterrupted/run.log" 2>&1
UNINTERRUPTED_RUN=$(find "$SMOKE_ROOT/uninterrupted/runs" -mindepth 1 \
  -maxdepth 1 -type d -name 'feature_retention_smoke_uninterrupted_*' | head -1)
test -f "$UNINTERRUPTED_RUN/results.json"
test -s "$UNINTERRUPTED_RUN/checkpoints/event_1_CIL.pt"

GPU=$(wait_for_gpu)
printf '[%s] interrupted smoke on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  "$PY" "${COMMON[@]}" --results_dir "$SMOKE_ROOT/resume/runs" \
  --exp_name feature_retention_smoke_resume \
  >"$SMOKE_ROOT/resume/first.log" 2>&1 &
SMOKE_PID=$!
RESUME_RUN=
while true; do
  RESUME_RUN=$(find "$SMOKE_ROOT/resume/runs" -mindepth 1 -maxdepth 1 \
    -type d -name 'feature_retention_smoke_resume_*' | head -1)
  if test -n "$RESUME_RUN" && \
      test -f "$RESUME_RUN/checkpoints/resume_latest.pt"; then
    break
  fi
  kill -0 "$SMOKE_PID"
  sleep 2
done
kill -TERM "$SMOKE_PID"
wait "$SMOKE_PID" || true
test -s "$RESUME_RUN/checkpoints/resume_latest.pt"

GPU=$(wait_for_gpu)
printf '[%s] resumed smoke on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  "$PY" "${COMMON[@]}" --results_dir "$SMOKE_ROOT/resume/runs" \
  --exp_name feature_retention_smoke_resume --resume_run_dir "$RESUME_RUN" \
  >"$SMOKE_ROOT/resume/resumed.log" 2>&1
test -f "$RESUME_RUN/results.json"
test -s "$RESUME_RUN/checkpoints/event_1_CIL.pt"

export UNINTERRUPTED_RUN RESUME_RUN
"$PY" - <<'PY'
import json
import os
from pathlib import Path

import torch

left_dir = Path(os.environ["UNINTERRUPTED_RUN"])
right_dir = Path(os.environ["RESUME_RUN"])
left = json.loads((left_dir / "results.json").read_text())
right = json.loads((right_dir / "results.json").read_text())
assert left["task_acc_history"] == right["task_acc_history"]
assert left["bic_history"] == right["bic_history"]
assert left["selection_audit"] == right["selection_audit"]


def compare(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left_item, right_item in zip(a, b):
            compare(left_item, right_item)
    else:
        assert a == b


left_ckpt = torch.load(
    left_dir / "checkpoints" / "event_1_CIL.pt", weights_only=False
)
right_ckpt = torch.load(
    right_dir / "checkpoints" / "event_1_CIL.pt", weights_only=False
)
compare(left_ckpt["trainer_state"], right_ckpt["trainer_state"])
compare(left_ckpt["cl_state"], right_ckpt["cl_state"])
(Path(os.environ["RESUME_RUN"]).parents[2] / "SMOKE_COMPARISON.json").write_text(
    json.dumps({
        "passed": True,
        "uninterrupted_run": str(left_dir),
        "resumed_run": str(right_dir),
        "task_acc_history_equal": True,
        "bic_history_equal": True,
        "selection_audit_equal": True,
        "checkpoint_state_equal": True,
    }, indent=2, sort_keys=True) + "\n"
)
PY
touch "$SMOKE_ROOT/SMOKE_SUCCESS"
trap - ERR
printf '[%s] smoke passed; starting formal staged study\n' \
  "$(date --iso-8601=seconds)"

exec env VFCL_FEATURE_ROOT="$STUDY_ROOT" \
  VFCL_REFERENCE_ROOT="$REFERENCE_ROOT" \
  VFCL_EXTERNAL_ROOT="$EXTERNAL_ROOT" \
  bash "$ROOT/run_cifar100_feature_retention_validation.sh"
