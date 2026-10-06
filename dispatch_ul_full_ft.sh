#!/usr/bin/env bash
# finetune x ALL 11 UL methods (UL-ONLY isolation baseline: no anti-forget CL,
# so UL behavior is not confounded by the CL method). Same o1 protocol,
# 3 seeds / 30 ep / P=2 / cosine head -> comparable to the proto_evolve UL table.
# QUEUED: waits for the proto_evolve UL batch (ALL_ULFULL_DONE.flag) to finish
# first so the two batches don't contend for the 8 DCUs. NB: no `set -u`.
cd /public/home/dongshou/cl_fix2/run
echo "waiting for proto_evolve UL batch to finish..."
while [ ! -f ALL_ULFULL_DONE.flag ]; do sleep 60; done
echo "proto_evolve batch done -> starting finetune x UL $(date)"
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--cl_method finetune --cosine_head \
  --num_parties 2 --seeds 42,43,44 --device cuda:0 --epochs_per_task 30 --ul_epochs 5 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5"

ULS=(retrain gradient_ascent luv mode fucrt fedup fedosd fedau fudp radapt_router roar)

rm -f ALL_ULFT_DONE.flag
i=0
for ul in "${ULS[@]}"; do
  gpu=$(( i % 8 ))
  out="./results/ul_full_ft/${ul}"
  echo "[GPU $gpu] finetune x $ul (3 seeds)"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py --ul_method "$ul" $COMMON \
    --results_dir "$out" > "log_ulft_${ul}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_ULFT_DONE.flag
echo "ALL UL FINETUNE DONE $(date)"
