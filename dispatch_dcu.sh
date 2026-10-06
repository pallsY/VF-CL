#!/bin/bash
# 8-DCU parallel dispatcher for per-method pure-CL verification on the cluster.
# Each JOB is "<tag>|<extra main.py args>"; jobs are pinned round-robin to
# HIP_VISIBLE_DEVICES 0..7 and run concurrently. Output -> results/_dcu/<tag>.
#
# Usage: bash dispatch_dcu.sh   (edit the JOBS array below)
set -u
cd /public/home/dongshou/projects/VF-CUL
source /opt/dtk-25.04.2/env.sh 2>/dev/null
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct

COMMON="--num_parties 1 --seeds 42 --device cuda:0 --epochs_per_task 30 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 99 --unlearn_classes 0 --cosine_head"

# tag|extra-args   (one per method/variant; <=8 run truly parallel)
JOBS=(
  "finetune|--cl_method finetune --ul_method luv"
  "lwf|--cl_method lwf --ul_method luv"
  "lwf_wa|--cl_method lwf_wa --ul_method luv"
  "ewc|--cl_method ewc --ul_method luv"
  "proto_evolve|--cl_method proto_evolve --ul_method luv"
  "prl|--cl_method prl --ul_method luv"
  "der_pp|--cl_method der_pp --ul_method luv"
  "er|--cl_method er --ul_method luv"
)

i=0
for job in "${JOBS[@]}"; do
  tag="${job%%|*}"; args="${job#*|}"
  gpu=$(( i % 8 ))
  out="./results/_dcu/${tag}"
  echo "[GPU $gpu] $tag :: $args"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $args $COMMON \
    --results_dir "$out" > "dcu_${tag}.log" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
echo "ALL DCU JOBS DONE"
