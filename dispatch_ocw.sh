#!/usr/bin/env bash
# CL leg headline: unlearning-aware CL (own_concentrate_weight) on the MAIN backbone
# (proto_evolve). Full continual timeline, forget the DIFFUSE class 4 (entropy 0.93)
# so |S*| shrinkage is visible. Measures UL benefit (|S*|,E|R_f|) vs CL cost (acc).
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh; source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct
COMMON="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz --model_type mlp \
  --num_parties 6 --aggregation concat --cosine_head --num_classes 7 \
  --custom_tasks 0,1|2,3|4,5|6 --cl_method proto_evolve --feat_distill_weight 1 \
  --ul_method roar --unlearn_after_tasks 2 --unlearn_classes 4 \
  --roar_tau_own 0.6 --roar_scrub_epochs 10 --roar_scrub_lr 1e-3 --roar_scrub_raw_weight 5 \
  --epochs_per_task 30 --seeds 42,43,44 --device cuda:0"
rm -f ALL_OCW_DONE.flag; i=0
for ocw in 0 0.25 0.5 1.0; do
  gpu=$(( i % 8 )); tag=$(echo $ocw|tr '.' 'p')
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $COMMON --own_concentrate_weight $ocw \
    --results_dir ./results/ocw_cl/ocw_${tag} > log_ocwcl_${tag}.txt 2>&1 </dev/null &
  i=$(( i+1 )); done
wait; touch ALL_OCW_DONE.flag; echo DONE
