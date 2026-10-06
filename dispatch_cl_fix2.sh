#!/usr/bin/env bash
# Re-run ewc + lwf_wa with the ported canonical-CE(new-only) + feature-KD fix.
# 8 jobs across 8 DCUs (round-robin HIP_VISIBLE_DEVICES 0..7), all concurrent.
# NB: deliberately NO `set -u` and NO `2>/dev/null` on env.sh -- the cluster
# env.sh references an unbound var, and `set -u` (esp. with stderr suppressed)
# makes the shell exit SILENTLY before any work runs (0-byte logs). This is the
# working pattern from dispatch_er.sh; dispatch_dcu.sh used set-u+2>/dev/null and
# produced no results.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

# Pure-CL c10 setup (no UL: unlearn_after_tasks=99 never matches a real task id).
COMMON="--num_parties 1 --seeds 42 --device cuda:0 --epochs_per_task 30 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 99 --unlearn_classes 0"

# tag|extra-args
JOBS=(
  "wa_cos_l1|--cl_method lwf_wa --ul_method luv --cosine_head --lwf_lambda 1 --feat_distill_weight 0"
  "wa_cos_l1_fkd1|--cl_method lwf_wa --ul_method luv --cosine_head --lwf_lambda 1 --feat_distill_weight 1"
  "wa_lin_l1|--cl_method lwf_wa --ul_method luv --lwf_lambda 1 --feat_distill_weight 0"
  "wa_lin_l1_fkd1|--cl_method lwf_wa --ul_method luv --lwf_lambda 1 --feat_distill_weight 1"
  "ewc_cos_l1k|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 1000 --feat_distill_weight 0"
  "ewc_cos_l100|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 100 --feat_distill_weight 0"
  "ewc_cos_fkd1_l100|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 100 --feat_distill_weight 1"
  "ewc_cos_fkd1_l0|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 0 --feat_distill_weight 1"
)

rm -f ALL_CLFIX2_DONE.flag
i=0
for job in "${JOBS[@]}"; do
  tag="${job%%|*}"; args="${job#*|}"
  gpu=$(( i % 8 ))
  out="./results/${tag}"
  echo "[GPU $gpu] $tag :: $args"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $args $COMMON \
    --results_dir "$out" > "log_${tag}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_CLFIX2_DONE.flag
echo "ALL cl_fix2 JOBS DONE $(date)"
