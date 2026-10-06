#!/bin/bash
# The 2 SLOW tin Oracles (tin_2p, tin_4p) — NOT recoverable from 40901 (never ran
# there). Each is ~10-20h (200 classes x 200 epochs x 3 seeds). Monopolizes the
# 2 shared cards for days, so launch this EXPLICITLY (not auto-queued).
# Idempotent. tin_1p oracle comes from the 40901 recovery, not here.
cd /public/home/dongshou/projects/VF-CUL
set +u; source /opt/dtk-25.04.2/env.sh; source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct; set -u
TIN='--data tinyimagenet --num_classes 200 --num_tasks 10 --classes_per_task 20 --unlearn_after_tasks 3,7 --unlearn_classes 20;100'
COMMON="--seeds 42,43,44 --device cuda:0 --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18"

run_oracle() {
  local card=$1 P=$2 OUT="./results/frag/tin_${P}p/oracle"
  if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then echo "[c$card] SKIP oracle tin-${P}p"; return; fi
  mkdir -p "$OUT"
  echo "[c$card] START oracle tin-${P}p $(date)"
  HIP_VISIBLE_DEVICES=$card CUDA_VISIBLE_DEVICES=$card python -u run_oracle_solo.py \
    --num_parties "$P" $COMMON $TIN --results_dir "$OUT"
  echo "[c$card] END oracle tin-${P}p $(date)"
}

mkdir -p logs_tin
run_oracle 6 2 > logs_tin/oracle_tin2p.log 2>&1 &
run_oracle 7 4 > logs_tin/oracle_tin4p.log 2>&1 &
wait
echo "TIN ORACLES DONE $(date)"
