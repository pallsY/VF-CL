#!/usr/bin/env bash
# Round 3: lwf_fim with the ported canonical new-class CE scope (+ additive KD).
# Genuine lwf_fim was still collapsed (~0.50) because it had the KD-scope + FIM
# fixes but NOT the new-class CE scope. Sweep FIM freeze frac + KD lambda + head.
# NB: no `set -u`; plain env.sh source.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--num_parties 1 --seeds 42 --device cuda:0 --epochs_per_task 30 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 99 --unlearn_classes 0"

JOBS=(
  "lwffim_cos_l1|--cl_method lwf_fim --ul_method luv --cosine_head --lwf_lambda 1 --fim_freeze_frac 0.25"
  "lwffim_cos_l1_f10|--cl_method lwf_fim --ul_method luv --cosine_head --lwf_lambda 1 --fim_freeze_frac 0.10"
  "lwffim_cos_l1_f40|--cl_method lwf_fim --ul_method luv --cosine_head --lwf_lambda 1 --fim_freeze_frac 0.40"
  "lwffim_cos_l2|--cl_method lwf_fim --ul_method luv --cosine_head --lwf_lambda 2 --fim_freeze_frac 0.25"
  "lwffim_cos_l3|--cl_method lwf_fim --ul_method luv --cosine_head --lwf_lambda 3 --fim_freeze_frac 0.25"
  "lwffim_lin_l1|--cl_method lwf_fim --ul_method luv --lwf_lambda 1 --fim_freeze_frac 0.25"
)

rm -f ALL_CLFIX2C_DONE.flag
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
touch ALL_CLFIX2C_DONE.flag
echo "ALL cl_fix2c JOBS DONE $(date)"
