#!/usr/bin/env bash
# UL smoke test on the FIXED CL code: proto_evolve (main, cosine+featKD) x ALL 11
# UL methods, c10, P=2, o1 unlearn protocol (after tasks 1,3 forget classes 0,5).
# Reduced epochs (15/task, ul_epochs 5) for a fast crash-catch + rough forget/
# retain/mia — ESPECIALLY for the never-run roar / radapt_router. 1 seed.
# Pinned to GPUs 3-7 so it doesn't collide with the multi-seed CL jobs on 0-2.
# NB: no `set -u`; plain env.sh source.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--cl_method proto_evolve --cosine_head --feat_distill_weight 1 \
  --num_parties 2 --seeds 42 --device cuda:0 --epochs_per_task 15 --ul_epochs 5 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5"

ULS=(retrain gradient_ascent luv mode fucrt fedup fedosd fedau fudp radapt_router roar)

rm -f ALL_ULSMOKE_DONE.flag
i=0
for ul in "${ULS[@]}"; do
  gpu=$(( i % 5 + 3 ))
  out="./results/ul_smoke/${ul}"
  echo "[GPU $gpu] proto_evolve x $ul"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py --ul_method "$ul" $COMMON \
    --results_dir "$out" > "log_ul_${ul}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_ULSMOKE_DONE.flag
echo "ALL UL SMOKE DONE $(date)"
