#!/usr/bin/env bash
# Multi-seed (42,43,44) confirmation of the 3 newly-fixed winners. main.py
# --seeds writes an aggregated.json with per-metric mean/std. NB: no `set -u`.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--num_parties 1 --seeds 42,43,44 --device cuda:0 --epochs_per_task 30 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 99 --unlearn_classes 0"

JOBS=(
  "wa_lin_l1_ms|--cl_method lwf_wa --ul_method luv --lwf_lambda 1 --feat_distill_weight 0"
  "lwffim_lin_l1_ms|--cl_method lwf_fim --ul_method luv --lwf_lambda 1 --fim_freeze_frac 0.25 --feat_distill_weight 0"
  "ewc_cos_l5k_ms|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 5000 --feat_distill_weight 0"
)

rm -f ALL_CLFIX2MS_DONE.flag
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
touch ALL_CLFIX2MS_DONE.flag
echo "ALL cl_fix2 MULTISEED DONE $(date)"
