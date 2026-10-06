#!/usr/bin/env bash
# Round 2 refinement. Round 1 showed: lwf_wa FIXED (wa_lin_l1=0.671, no featKD);
# ewc improved (ewc_cos_l1k=0.499) but early tasks fade; featKD=1 way too strong
# (collapses to task 0). Here: EWC lambda-sweep + small featKD; confirm lwf_wa.
# NB: no `set -u`; plain env.sh source (see dispatch_cl_fix2.sh header).
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--num_parties 1 --seeds 42 --device cuda:0 --epochs_per_task 30 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 99 --unlearn_classes 0"

JOBS=(
  "ewc_cos_l2k|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 2000 --feat_distill_weight 0"
  "ewc_cos_l5k|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 5000 --feat_distill_weight 0"
  "ewc_cos_l10k|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 10000 --feat_distill_weight 0"
  "ewc_cos_l1k_fkd005|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 1000 --feat_distill_weight 0.05"
  "ewc_cos_l1k_fkd02|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 1000 --feat_distill_weight 0.2"
  "ewc_cos_fkd02_l0|--cl_method ewc --ul_method luv --cosine_head --ewc_lambda 0 --feat_distill_weight 0.2"
  "wa_lin_l2|--cl_method lwf_wa --ul_method luv --lwf_lambda 2 --feat_distill_weight 0"
  "wa_lin_l1_fkd01|--cl_method lwf_wa --ul_method luv --lwf_lambda 1 --feat_distill_weight 0.1"
)

rm -f ALL_CLFIX2B_DONE.flag
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
touch ALL_CLFIX2B_DONE.flag
echo "ALL cl_fix2b JOBS DONE $(date)"
