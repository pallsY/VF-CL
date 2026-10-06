#!/usr/bin/env bash
# Unified VFL continual+unlearning timeline: ALTERNATING learn-task / forget-class.
# learn 0,1 | learn 2,3 | FORGET 0 | learn 4,5 | FORGET 2 | learn 6 | FORGET 4.
# Compare vanilla CL (ocw=0) vs unlearning-aware CL (ocw=0.5), proto_evolve, 3 seeds.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh; source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct
COMMON="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz --model_type mlp \
  --num_parties 6 --aggregation concat --cosine_head --num_classes 7 \
  --custom_tasks 0,1|2,3|4,5|6 --cl_method proto_evolve --feat_distill_weight 1 \
  --ul_method roar --unlearn_after_tasks 1,2,3 --unlearn_classes 0;2;4 \
  --roar_tau_own 0.6 --roar_scrub_epochs 10 --roar_scrub_lr 1e-3 --roar_scrub_raw_weight 5 \
  --epochs_per_task 30 --seeds 42,43,44 --device cuda:0"
rm -f ALL_TL_DONE.flag; i=0
for ocw in 0 0.5; do
  gpu=$(( i % 8 )); tag=$(echo $ocw|tr '.' 'p')
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $COMMON --own_concentrate_weight $ocw \
    --results_dir ./results/timeline/ocw_${tag} > log_tl_${tag}.txt 2>&1 </dev/null &
  i=$(( i+1 )); done
wait; touch ALL_TL_DONE.flag; echo DONE
